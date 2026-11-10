"""Deterministic checks with generated inputs only; no native HPP reader calls."""
import copy
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import native
from demo_data import make
from frozen import modules
from functional import Design, FixedMemberTarget, FunctionState, TechnicalState
from pilot import checkpoint, execute, outer, paired_intervals, report, save_arrays, sha, summarize

ROOT=Path(__file__).resolve().parent


def config():
    c=json.loads((ROOT/'config.json').read_text())
    c.update(outer_folds=3,inner_folds=2,prediction_bootstraps=99,functional_main_components=5,
             main_penalties=[.01,.1],interaction_penalties=[.03,.3],support_neighbors=40)
    return c


def expect_error(action, fragment):
    try:
        action()
    except (ValueError,RuntimeError) as error:
        assert fragment in str(error),(fragment,str(error))
    else:
        raise AssertionError('Expected rejection: '+fragment)


def check_sequential_archive(passed):
    original_open=io.open
    writers=[]
    class ForwardOnly:
        def __init__(self,stream,fail_write=False):
            self.stream,self.fail_write=stream,fail_write
            self.seeks=[]
            self.backward_rejected=False
        def __getattr__(self,name):
            return getattr(self.stream,name)
        def __enter__(self):
            return self
        def __exit__(self,*args):
            self.stream.close()
        def seek(self,offset,whence=0):
            self.seeks.append((offset,whence))
            target=offset if whence==0 else self.stream.tell()+offset if whence==1 else os.fstat(self.stream.fileno()).st_size+offset
            if target < self.stream.tell() and not self.backward_rejected:
                # Permit later destructor cleanup after reproducing the first
                # unsupported backwards seek, avoiding ignored-finalizer noise.
                self.backward_rejected=True
                raise OSError(95,'Simulated unsupported backwards write seek')
            return self.stream.seek(offset,whence)
        def write(self,payload):
            if self.fail_write:
                self.stream.write(payload[:5])
                raise OSError(5,'Simulated interrupted sequential write')
            return self.stream.write(payload)
    def hook(fail_write=False):
        def opened(path,mode='r',*args,**kwargs):
            result=original_open(path,mode,*args,**kwargs)
            if isinstance(path,(str,os.PathLike)) and str(path).endswith('.npz') and any(c in mode for c in 'wxa'):
                result=ForwardOnly(result,fail_write)
                writers.append(result)
            return result
        return opened
    with tempfile.TemporaryDirectory(prefix='hpp_npz_forward_only_synthetic_') as tmp:
        folder=Path(tmp)
        arrays=dict(flags=np.array([True,False,True]),values=np.arange(18,dtype=float).reshape(6,3))
        reproduced=False
        with patch('io.open',hook()):
            try:
                np.savez_compressed(str(folder/'old_direct.npz'),**arrays)
            except OSError as error:
                reproduced=error.errno==95
        passed('original_path_zip_writer_reproduces_errno95',reproduced and any(w.backward_rejected for w in writers))
        writers.clear()
        target=folder/'new_sequential.npz'
        with patch('io.open',hook()):
            save_arrays(target,arrays)
        passed('new_archive_writer_never_seeks_target_file',bool(writers) and all(not w.seeks for w in writers))
        with np.load(str(target),allow_pickle=False) as loaded:
            passed('sequential_npz_roundtrip_preserves_flags_values_and_shapes',all(np.array_equal(loaded[name],value) and loaded[name].dtype==value.dtype for name,value in arrays.items()))
        before=target.read_bytes()
        failed=False
        with patch('io.open',hook(True)):
            try:
                save_arrays(target,dict(flags=np.array([False])))
            except OSError as error:
                failed=error.errno==5
        passed('interrupted_write_preserves_completed_file_and_cleans_temporary',failed and target.read_bytes()==before and not list(folder.glob('new_sequential.tmp-*')))


def native_adapter_fixture():
    """Run the copied source audit on generated files, mock only the old model reader."""
    import demo_files as fixture
    with tempfile.TemporaryDirectory(prefix='hpp_function_adapter_synthetic_') as tmp:
        state=fixture.fixture(Path(tmp))
        for which,path in enumerate(state['paths']):
            with pa.memory_map(str(path),'r') as stream:
                table=ipc.open_file(stream).read_all()
            rows=table.to_pylist()
            for row in rows:
                if row['participant_id']==101:
                    for name in ('PWY-1234:demo','GLYCOLYSIS-PWY:demo','VALID-OTHER-FEATURE'):
                        row[name]=None
                if row['participant_id']==102:
                    row['GLYCOLYSIS-PWY:demo']=None
            result=pa.Table.from_pylist(rows,schema=table.schema)
            with pa.OSFile(str(path),'wb') as out:
                with ipc.new_file(out,result.schema) as writer:
                    writer.write_table(result)
        index=pq.ParquetFile(state['cov']).read().to_pandas(ignore_metadata=True)
        with np.load(str(state['reference']),allow_pickle=False) as archive:
            reference=dict(y=archive['y'])
        dataset=SimpleNamespace(index=index,reference=reference,paths=[state['cov']],y=np.arange(5,dtype=float),api=SimpleNamespace(PhenotypeOutliers=lambda:None))
        frozen=modules()
        gate=Path(tmp)/'existing_summary.json'
        gate.write_text('{"synthetic_only":true}')
        fake=SimpleNamespace(root=frozen.root,config=frozen.config,upload_check=frozen.upload_check,
            runtime=SimpleNamespace(environment=lambda c:dict(synthetic_only=True),readiness=lambda *a,**k:dict(v11_summary=str(gate))),
            bridge=SimpleNamespace(load=lambda *a:dataset,adapter_sources=lambda:[]))
        before=native.modules
        old_handler=signal.getsignal(signal.SIGALRM)
        native.modules=lambda:fake
        try:
            cfg=config()
            cfg.update(original_people=3,functional_people=2,candidate_fields=3,source_input_rows=5,original_households=2)
            data=native.load(state['project'],state['raw'],cfg)
        finally:
            native.modules=before
        assert signal.getsignal(signal.SIGALRM)==old_handler
        assert np.flatnonzero(data.available).tolist()==[1,2]
        assert np.isnan(data.signals['abundance'][0]).all()
        assert data.signals['abundance'][1,1] == 2.  # sorted public labels: glycolysis, PWY, other
        assert data.audit['excluded_whole_candidate_null']==1
        assert 'PATHWAY_NULL_CONTRACT_DONE' in data.audit_text
        assert not data.synthetic_only  # native adapter path was exercised, all files were generated


def main():
    checks=[]
    def passed(name, condition=True):
        assert condition,name
        checks.append(name)
    cfg=config()
    check_sequential_archive(passed)
    class OriginalTarget:
        def fit(self,y,participant_ids,*,endpoint,role='training'):
            self.state=dict(kind='synthetic_original',clip_low=-5.,clip_high=5.)
            return self
        def transform(self,y,*,endpoint):
            a=np.asarray(y,float)
            return dict(values=np.where(abs(a)>8,np.nan,np.clip(a,-5,5)),excluded_8sd=abs(a)>8)
    target=FixedMemberTarget(OriginalTarget()).fit(np.array([-1.,0.,1.]),[0,1,2],endpoint='D_Glu')
    transformed=target.transform(np.array([100.,np.nan,-100.,1.]),endpoint='D_Glu')
    passed('fixed_eligible_extreme_outcomes_clipped_not_rescreened',np.array_equal(transformed['values'][[0,2,3]],np.array([5.,-5.,1.])) and not transformed['excluded_8sd'].any())
    passed('nonfinite_outcomes_not_manufactured',np.isnan(transformed['values'][1]) and transformed['original_rule_excluded_8sd'][0] and transformed['winsorized_high'][0] and transformed['winsorized_low'][2])
    data=make(cfg)
    frozen=modules()
    a=data.signals['abundance'][:300]
    state=FunctionState('abundance_relative',cfg).fit(a)
    multiplier=np.linspace(.01,100,len(a))[:,None]
    passed('relative_signal_invariant_to_positive_sample_scale',np.allclose(state.transform(a),state.transform(a*multiplier),atol=1e-10))
    ranks=FunctionState('abundance_unknown_null',cfg).fit(a)
    b=a.copy()
    b[:,0]=np.nan
    passed('unknown_null_encoding_retains_reporting_information',not np.allclose(ranks.raw(a),ranks.raw(b)))
    rank_scaled=FunctionState('abundance_unknown_null',cfg).fit(a*np.arange(1,a.shape[1]+1))
    passed('observed_rank_representation_invariant_to_feature_units',np.allclose(ranks.raw(a),rank_scaled.raw(a*np.arange(1,a.shape[1]+1))))
    empty=np.ones((100,5))
    constant=FunctionState('abundance_relative',cfg).fit(empty)
    passed('constant_relative_spectrum_has_zero_components',constant.transform(empty).shape==(100,0))
    never=np.full((100,4),np.nan)
    never[:,0]=1.
    rare=FunctionState('coverage_confidence',cfg).fit(never)
    passed('training_missing_fields_not_selected',rare.keep.tolist()==[0])
    expect_error(lambda:state.transform(np.full((2,a.shape[1]),np.nan)),'flagged')
    passed('entire_null_profile_not_made_biological_zero')
    expect_error(lambda:FunctionState('coverage_confidence',cfg).fit(np.full((50,4),2.)),'range')
    passed('coverage_range_checked')
    tech=TechnicalState().fit(data.technical[:300])
    old_median=tech.median.copy()
    transformed=tech.transform(np.full((5,3),np.nan))
    passed('technical_missing_values_train_fitted_and_finite',np.array_equal(tech.median,old_median) and np.isfinite(transformed).all())
    train=data.available & (data.dataset.reference['outer'][0] != 0)
    valid=data.available & ~train
    base=data.frozen.engine.Engine(None,None).design(data.dataset,train,valid)
    view=Design(base,'abundance_relative',data.signals['abundance'],data.technical,cfg,frozen.features)
    spec=dict(name='abundance_relative')
    zt,zv,imap=view.interactions(spec)
    passed('functional_interaction_residualized_against_matched_main',np.max(abs(view.xt.T@zt))<1e-6)
    baseline=frozen.models.fit_main(view,.1)
    no_interaction=frozen.models.fit_augmented(view,spec,.1,None,baseline)
    passed('no_interaction_is_exact_same_fitted_baseline',no_interaction is baseline and np.array_equal(no_interaction.predict(),baseline.predict()))
    fit=frozen.models.fit_augmented(view,spec,.1,.03)
    total,component=fit.contrasts()
    passed('counterfactual_model_contrast_matches_direct_prediction_difference',np.allclose(total,fit.predict(diet=15)-fit.predict(diet=10)))
    passed('main_effect_contains_all_interaction_pc_coordinates',view.pcs[0].shape[1]>=zt.shape[1] and np.array_equal(view.xt[:,0],np.ones(len(view.xt))))
    passed('functional_nonlinear_main_terms_in_both_models',view.function_curves is not None and len(view.function_curves.keep)>0 and baseline.design is fit.design)
    unchanged=state.pca.components_.copy()
    state.transform(np.where(np.isnan(a[:5]),np.nan,a[:5]*100000))
    passed('heldout_function_values_do_not_refit_pca',np.array_equal(state.pca.components_,unchanged))

    first,meta=outer(data,cfg,0,0,lambda *a:None)
    altered=make(cfg)
    outer_valid=altered.available & (altered.dataset.reference['outer'][0]==0)
    altered.dataset.y[outer_valid] += 1000000.
    second,meta2=outer(altered,cfg,0,0,lambda *a:None)
    passed('outer_validation_outcome_does_not_change_selection',meta['selections']==meta2['selections'])
    passed('outer_validation_outcome_does_not_change_predictions',np.array_equal(first['main'],second['main']) and np.array_equal(first['interaction'],second['interaction']))
    passed('outer_outcome_mutation_reaches_only_evaluation_target',not np.array_equal(first['y'],second['y']))
    no_variation=make(cfg)
    for values in no_variation.signals.values():
        values[no_variation.available]=1.
    same_prediction,constant_meta=outer(no_variation,cfg,0,0,lambda *a:None)
    passed('zero_dimensional_functions_exactly_revert_to_main_in_nested_selection',np.array_equal(same_prediction['main'],same_prediction['interaction']) and all(v['interaction_penalty'] is None for v in constant_meta['selections'].values()))
    native_adapter_fixture()
    passed('native_adapter_full_generated_file_lineage_and_row_mapping')
    with tempfile.TemporaryDirectory(prefix='hpp_function_pilot_synthetic_') as tmp:
        folder=Path(tmp)
        failed_run=folder/'failed_before_models'
        failure_logs=[]
        startup_failure=False
        with patch('pilot.save_arrays',side_effect=OSError(95,'Simulated old startup failure')):
            try:
                execute(data,cfg,failed_run,threads=1,log=failure_logs.append)
            except OSError as error:
                startup_failure=error.errno==95
        preserved={name:(failed_run/name).read_bytes() for name in ('registration.json','RUNNING.lock','source_audit.txt')}
        passed('cohort_write_failure_occurs_before_any_model_or_complete_fold',startup_failure and not failure_logs and not list((failed_run/'TRE_ONLY').glob('*_COMPLETE.json')) and not (failed_run/'summary.json').exists())
        summary=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('separate_repaired_run_preserves_failed_registration_and_lock',all((failed_run/name).read_bytes()==value for name,value in preserved.items()) and summary['status']=='COMPLETE')
        primary=[r for r in summary['scores'] if r['route']==cfg['primary_route']]
        passed('known_synthetic_interaction_gains_on_both_household_splits',all(r['delta_R2']>0 for r in primary))
        passed('all_routes_and_splits_reported',len(summary['scores'])==6 and all(r['N']==420 for r in summary['scores']))
        passed('synthetic_results_explicitly_labeled',summary['synthetic_only'] is True)
        passed('six_fold_atomic_completion_receipts',len(list((folder/'run/TRE_ONLY').glob('*_COMPLETE.json')))==6)
        again=execute(data,cfg,folder/'run',threads=1,log=lambda *a:None)
        passed('completed_run_replayed_without_refit',again==summary)
        tampered=folder/'run/TRE_ONLY/final_predictions.npz'
        with tampered.open('ab') as stream:
            stream.write(b'changed')
        expect_error(lambda:execute(data,cfg,folder/'run',threads=1,log=lambda *a:None),'checksum')
        passed('tampered_completed_arrays_rejected')
        ck=folder/'chunks'
        ck.mkdir()
        arrays=dict(rows=np.array([1]),y=np.array([2.]))
        checkpoint(ck,'one','same',lambda:(arrays,dict(synthetic_only=True)))
        expect_error(lambda:checkpoint(ck,'one','changed',lambda:None),'identity')
        passed('changed_checkpoint_identity_rejected')
        (ck/'one.json').write_text('{}')
        expect_error(lambda:checkpoint(ck,'one','same',lambda:None),'checksum')
        passed('tampered_fold_metadata_rejected')
        preview=[]
        from contextlib import redirect_stdout
        import io
        output=io.StringIO()
        with redirect_stdout(output):
            report(summary)
        preview=output.getvalue()
        passed('report_contains_scope_and_no_individual_identifiers','SYNTHETIC ONLY' in preview and 'SYNTHETIC_HH_' not in preview and 'formal interaction P value' in preview)
    # Explicitly adverse predictions: every planned negative result must survive aggregation.
    chunks=[]
    y=data.dataset.y
    for repeat in range(2):
        for fold in range(cfg['outer_folds']):
            rows=np.flatnonzero(data.available & (data.dataset.reference['outer'][repeat]==fold))
            main=np.tile(y[rows],(3,1))
            arrays=dict(rows=rows,y=y[rows],raw_y=y[rows],main=main,interaction=main+1.,total_contrast=np.zeros_like(main),delta_contrast=np.zeros_like(main),raw_component=np.zeros_like(main),support=np.ones_like(main,dtype=bool))
            chunks.append((arrays,dict(repeat=repeat,fold=fold,selections={route:dict(interaction_penalty=.03) for route in cfg['routes']})))
    negative,_=summarize(data,cfg,chunks)
    passed('negative_incremental_R2_preserved_for_every_route',all(row['delta_R2']<0 for row in negative['scores']))
    passed('constant_split_contrast_correlation_is_missing_not_zero',all(v is None for v in negative['same_people_added_contrast_split_correlation'].values()))
    error_chunks=chunks+[chunks[0]]
    expect_error(lambda:summarize(data,cfg,error_chunks),'duplicated')
    passed('duplicate_outer_members_rejected')
    groups=data.dataset.index.loc[data.available,'split_unit_id'].astype(str).to_numpy()
    noise=np.tile(y[data.available],(3,1))
    interval=paired_intervals(y[data.available],noise,noise+1.,groups,99,7)
    passed('paired_household_interval_has_correct_adverse_direction',np.all(interval<0))
    for p in ROOT.glob('*.py'):
        compile(p.read_text(),str(p),'exec')
    passed('all_python_sources_compile')
    destination=ROOT/'research_records/local_validation'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'synthetic_preview.txt').write_text('GENERATED DATA ONLY; NOT HPP FINDINGS\n\n'+preview)
    result=dict(status='passed',synthetic_only=True,check_count=len(checks),checks=checks,native_HPP_inputs_read=False,
        native_environment_execution_verified=False,scope='Generated matrices/files; frozen source imports; native reader exercised on generated fixtures only.')
    (destination/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FUNCTIONAL_SYNTHETIC_CHECKS_OK checks=%d; no native HPP data read.' % len(checks))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
