"""W18 technical preprocessing reader: sealed roles, explicit fields, receipts.

This workflow guard permits technical transforms only. It never authorizes
association models, supervised score learning, or confirmation inference.
"""
from pathlib import Path
import hashlib,json,datetime
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
K=['participant_id','cohort','research_stage']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class TechnicalPreprocessingReader:
    def __init__(self,receipt_directory,requests):
        root=P/'config/preprocessing_20261030_v1'
        self.pointer=json.loads((root/'active_execution.json').read_text())
        assert sha(self.pointer['execution_design'])==self.pointer['execution_design_sha256']
        assert sha(self.pointer['roles'])==self.pointer['roles_sha256']
        self.design=json.loads(Path(self.pointer['execution_design']).read_text())
        self.roles=pq.ParquetFile(self.pointer['roles'],pre_buffer=False).read(columns=K+['split_unit_id','final_role'],use_threads=False).to_pandas(ignore_metadata=True)
        assert not self.roles.duplicated(K).any()
        assert set(self.roles.final_role)=={'discovery','confirmation_reserved'}
        assert self.roles.groupby('split_unit_id').final_role.nunique().max()==1
        self.receipts=Path(receipt_directory);self.receipts.mkdir(exist_ok=False)
        self.requests={str(Path(v['path'])):v for v in requests}
        assert len(self.requests)==len(requests)
        (self.receipts/'allowed_reads.json').write_text(json.dumps({'requests':requests,'roles_sha256':self.pointer['roles_sha256'],
            'design_sha256':self.pointer['execution_design_sha256'],'association_or_supervised_fit_permitted':False},indent=2)+'\n')
        self.count=0

    def read(self,path,columns,purpose):
        path=Path(path);key=str(path)
        request=self.requests[key]
        assert path.is_relative_to(P) and purpose==request['purpose']
        assert set(columns)<=set(request['columns']) and set(K)<=set(columns)
        assert purpose in {'microbiome_transform','phenotype_QC','covariate_encoding','food_inputs'}
        if purpose=='phenotype_QC':
            assert set(columns)<=set(K+['Y_value','base_eligible','Y_unit','Y_valid_measurements','n_qualified_CGM_days'])
        pf=pq.ParquetFile(path,pre_buffer=False)
        assert set(columns)<=set(pf.schema_arrow.names)
        # Full requested projections are budgeted, then restricted by sealed IDs.
        assert pf.metadata.num_rows*max(1,len(columns))*16<1024**3
        self.count+=1
        receipt={'sequence':self.count,'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'path':key,'columns':columns,'purpose':purpose,'scientific_Y_values':purpose=='phenotype_QC',
                 'source_sha256':sha(path),'source_rows':pf.metadata.num_rows,'roles_sha256':self.pointer['roles_sha256'],
                 'fit_role':'discovery_only','confirmation_role':'apply_frozen_transform_only',
                 'association_or_supervised_fit_permitted':False}
        target=self.receipts/(str(self.count).zfill(3)+'_preaccess.json')
        with target.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        filt=(ds.field('cohort')=='10k')&(ds.field('research_stage')=='00_00_visit')&ds.field('participant_id').isin(self.roles.participant_id.tolist())
        table=ds.dataset(path,format='parquet').scanner(columns=columns,filter=filt,use_threads=False,batch_size=1024).to_table()
        d=table.to_pandas(ignore_metadata=True)
        assert not d.duplicated(K).any(),'source is not one row per person/visit'
        d=d.merge(self.roles,on=K,how='inner',validate='one_to_one')
        assert d.final_role.notna().all()
        return d

    def training(self,d):
        t=d.loc[d.final_role.eq('discovery')].copy()
        reserved=set(self.roles.loc[self.roles.final_role.eq('confirmation_reserved'),'participant_id'])
        assert not set(t.participant_id)&reserved
        assert not t.duplicated(K).any()
        return t
