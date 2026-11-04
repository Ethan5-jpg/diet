import numpy as np
from features import pcs, modules, backmap

def test():
    rng=np.random.default_rng(14)
    logs=rng.normal(size=(140,26))
    logs[:,:5]=logs[:,[0]]+rng.normal(scale=.05,size=(140,5))
    tr=np.arange(len(logs))<90
    clr=logs-logs.mean(1,keepdims=True)
    # Exercise a reference species omitted from output, plus nonunit scales.
    out=np.arange(1,26)
    center=clr[tr][:,out].mean(0)
    scale=clr[tr][:,out].std(0)
    z=(clr[:,out]-center)/scale
    state=dict(scales=scale,means=center,reference_indices=np.arange(26),output_indices=out)
    pc,p=pcs(z,tr); coef=rng.normal(size=10)
    offset=(pc[tr]@coef).mean()
    _,b,c=backmap(state,p,coef,offset)
    np.testing.assert_allclose(logs@b+c,pc@coef-offset,atol=1e-12)
    assert abs(b.sum())<1e-12
    m,ms=modules(z,state,tr)
    assert any(set(range(4))<=set(x) for x in ms)
    changed=z.copy();changed[~tr]=50
    p2,_=pcs(changed,tr);m2,ms2=modules(changed,state,tr)
    np.testing.assert_allclose(p2[tr],pc[tr]);np.testing.assert_allclose(m2[tr],m[tr])
    assert [list(x) for x in ms]==[list(x) for x in ms2]
if __name__=='__main__':
    test(); print('SYNTHETIC_PASS')
