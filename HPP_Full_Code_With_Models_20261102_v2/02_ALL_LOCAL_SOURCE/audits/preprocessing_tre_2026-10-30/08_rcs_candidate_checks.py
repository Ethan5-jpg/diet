from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from w17_interval_upf_rcs import rcs3,design_ranges,moment_ranges
Q=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/data_qc/preprocessing_20261030_v1')
r=np.random.default_rng(20261030008)
# Independent cubic interpolation with zero second derivative at the outer knots.
for knots in [[-3.2,0,3.2],[-1.28,0,1.28],[-4,1,2],[0,.1,9]]:
    k=np.asarray(knots);a,b=k[1]-k[0],k[2]-k[1];span=a+b
    values=[0,a**3/span**2,a*(a+2*b)/span]
    cs=CubicSpline(k,values,bc_type='natural')
    x=np.linspace(k[0],k[2],1001)
    np.testing.assert_allclose(rcs3(x,k),cs(x),atol=1e-10)
    assert np.min(np.diff(rcs3(np.linspace(k[0]-100,k[2]+100,10001),k)))>=-1e-8
    np.testing.assert_allclose(rcs3(k[2]+np.array([0,1,2]),k),values[-1]+3*a/span*np.array([0,1,2]))
lo=r.uniform(0,.7,300);hi=lo+r.uniform(0,.3,300);c=r.normal(size=(300,2));s=r.normal(size=(300,2));y=r.normal(size=300)
xl,xu,idx=design_ranges(lo,hi,c,s,center=.5,delta=.1)
ml,mu,pairs=moment_ranges(xl,xu,y)
for i in range(100):
    d=lo+(hi-lo)*r.random(300)
    xd,_,_=design_ranges(d,d,c,s,center=.5,delta=.1)
    assert np.all(xd>=xl-1e-10) and np.all(xd<=xu+1e-10)
    md,_,_=moment_ranges(xd,xd,y)
    assert np.all(md>=ml-1e-8) and np.all(md<=mu+1e-8)
report={'status':'candidate_RCS_dictionary_checks_passed','independent_scipy_natural_cubic_comparison':True,
        'monotonic_range_and_linear_tail_verified':True,'latent_design_moment_containment_draws':100,
        'statistical_calibration':'pending','new_scientific_Y_reads':0}
(Q/'RCS_candidate_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('RCS_CANDIDATE_PASS',json.dumps(report),flush=True)
