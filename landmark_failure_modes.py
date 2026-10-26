"""Frozen clean models: independent jitter, missing-point and outlier stress tests."""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits
from landmark_robustness import perturb
from landmark_experiment import CLASSES, EDGES
from multiwidth_experiment import read, save, snapshot, metrics

JITTER=[0,.01,.025,.05,.1,.2,.3,.5,.7]
COUNTS=[0,1,2,4,6,10,14]
NAMES=['A_linear','B_rbf32','K_full_gaussian']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_failure_modes_v1'));out=ap.parse_args().out
    out.mkdir(parents=True,exist_ok=False);base=Path('artifacts/hagrid_landmarks_v1');before=snapshot(base)
    save(out/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),jitter=JITTER,counts=COUNTS,seeds=list(range(10)),
        missing='In clean normalized coordinates, replace selected non-wrist points by inner-training class-agnostic coordinate means. No re-normalization. Assumes clean scale available; controlled diagnostic only.',
        outlier='Select non-wrist points; displace each by exactly 1 clean RMS hand scale in a uniform random direction; re-normalize using corrupted points. Wrist unchanged.',
        jitter_definition='Independent Gaussian x/y noise including wrist, SD relative to clean RMS; renormalize as prior experiment.',
        selection='None. Frozen clean models and original scaler/prototypes. Same corruption for all models. No test data. Nested masks across counts.',
        multiwidth='Not separately reported: previous clean-validation multiwidth search yielded identical single-width configurations.',
        limitation='Exploratory reused development users; synthetic failures, no hardware measurements.'))
    save(out/'source_hashes.json',before);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(base/'versions.json',out/'versions.json')
    m=pd.read_csv(base/'development_manifest.csv');m.to_csv(out/'manifest.csv',index=False)
    x=np.load(base/'landmark_features.npz',allow_pickle=True)['features'];assert m.split.ne('test').all()
    bundles=[];imputed=np.zeros_like(x);fold_of=np.full(len(x),-1)
    for fold in range(3):
        p=base/f'fold_{fold}';ids=read(p/'split_indices.json');a,b,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
        users=[set(m.iloc[v].user_id) for v in [a,b,c]];assert not(users[0]&users[1] or users[0]&users[2] or users[1]&users[2])
        shared=joblib.load(p/'shared.joblib');assert np.allclose(shared['scaler'].mean_,x[a].mean(axis=0))
        assert (fold_of[c]==-1).all();fold_of[c]=fold;imputed[c]=x[a].mean(axis=0)
        bundles.append((shared,{n:joblib.load(p/(n+'.joblib')) for n in NAMES},c))
        save(out/f'fold_{fold}_audit.json',dict(indices=ids,imputer_train_mean=x[a].mean(axis=0).tolist()))
    assert (fold_of>=0).all()
    rows=[];frames=[];classes=[];responses=[];previews={}
    with threadpool_limits(limits=1):
        for seed in range(10):
            noise=np.random.default_rng(seed).standard_normal((len(x),21,2))
            rng=np.random.default_rng(1000+seed);order=np.argsort(rng.random((len(x),20)),axis=1)+1
            angles=np.random.default_rng(2000+seed).uniform(0,2*np.pi,(len(x),21))
            shifts=np.stack([np.cos(angles),np.sin(angles)],axis=-1)
            variants=[];keys=[]
            for kind,levels in [('jitter',JITTER),('missing',COUNTS),('outlier',COUNTS)]:
                for level in levels:
                    if kind=='jitter':xx=perturb(x,noise,level)
                    else:
                        pts=x.reshape(-1,21,2).copy();sample=np.arange(len(x))[:,None];indices=order[:,:level]
                        if kind=='missing':pts[sample,indices]=imputed.reshape(-1,21,2)[sample,indices]
                        else:
                            pts[sample,indices]+=shifts[sample,indices]
                            if level:pts/=np.sqrt(np.mean(np.sum(pts[:,1:]**2,axis=2),axis=1))[:,None,None]
                        xx=pts.reshape(-1,42)
                    assert np.isfinite(xx).all() and np.allclose(xx[:,:2],0)
                    if level==0:assert np.array_equal(xx,x)
                    variants.append(xx);keys.append(f'{kind}_{level}')
                    if seed==0 and level in ([0,.2,.5,.7] if kind=='jitter' else [0,4,10,14]):previews[(kind,level)]=xx.copy()
                    predictions={n:np.full(len(x),'',dtype=object) for n in NAMES}
                    for fold,(shared,models,idx) in enumerate(bundles):
                        z=shared['scaler'].transform(xx[idx]);centers=shared['kmeans'].cluster_centers_
                        for name,bundle in models.items():
                            model=bundle['model']
                            if name=='A_linear':pred=model.predict(z)
                            elif name=='B_rbf32':
                                d2=np.sum((z[:,None]-centers[None])**2,axis=2);ft=np.exp(-d2/(2*bundle['config']['sigma']**2))
                                assert np.isfinite(ft).all()
                                responses.append(dict(seed=seed,kind=kind,level=level,fold=fold,all_zero_rows=int(np.all(ft==0,axis=1).sum()),fraction_below_1e8=float((ft<1e-8).mean())))
                                pred=model.predict(ft)
                            else:
                                e=bundle['extra'];k=e['centerer'].transform(rbf_kernel(z,e['train_features'],gamma=e['gamma']))
                                pred=np.array(CLASSES)[(model.predict(k)+e['target_mean']).argmax(axis=1)]
                            if level==0:
                                old=pd.read_csv(base/f'fold_{fold}'/(name+'_outer_predictions.csv'))
                                assert np.array_equal(old.sample_id,m.iloc[idx].sample_id) and np.array_equal(old.predicted_label,pred)
                            predictions[name][idx]=pred
                    for name,pred in predictions.items():
                        met=metrics(m['class'],pred);tag=dict(seed=seed,kind=kind,level=level,model=name)
                        rows.append(dict(**tag,**{k:met[k] for k in ['accuracy','macro_f1','balanced_accuracy']}))
                        for cls in CLASSES:classes.append(dict(**tag,label=cls,**met['per_class'][cls]))
                        save(out/'metrics'/f'{kind}_{level}_{seed}_{name}.json',met)
                        f=m[['sample_id','user_id','class']].copy();f['predicted_label']=pred
                        for k,v in tag.items():f[k]=v
                        frames.append(f)
            np.savez_compressed(out/f'corruptions_seed_{seed}.npz',features=np.array(variants),keys=np.array(keys),sample_ids=m.sample_id.to_numpy(dtype=str),point_order=order,unit_noise=noise,angles=angles,fold=fold_of)
            print('FINISHED SEED',seed,flush=True)
    pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(classes).to_csv(out/'per_class.csv',index=False);pd.DataFrame(responses).to_csv(out/'rbf_response_audit.csv',index=False)
    r=pd.DataFrame(rows);r.to_csv(out/'seed_metrics.csv',index=False)
    s=r.groupby(['kind','level','model']).agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),noise_seed_std=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index()
    s.to_csv(out/'summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,kind in zip(axes,['jitter','missing','outlier']):
        for name in NAMES:
            q=s[(s.kind==kind)&(s.model==name)];ax.errorbar(q.level*(100 if kind=='jitter' else 1),q.macro_f1_mean,yerr=q.noise_seed_std,marker='o',label=name)
        ax.set(title=kind,xlabel='Noise SD / clean scale (%)' if kind=='jitter' else 'Affected non-wrist points (out of 20)',ylabel='Macro-F1',ylim=(0,1.03));ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'failure_curves.png',dpi=160);plt.close(fig)
    for kind in ['jitter','missing','outlier']:
        selected=[(k,v) for k,v in previews.items() if k[0]==kind];fig,axes=plt.subplots(4,len(selected),figsize=(12,10))
        for row,cls in enumerate(CLASSES):
            idx=int(np.flatnonzero(m['class'].eq(cls))[0])
            for col,((_,level),xx) in enumerate(selected):
                ax=axes[row,col];p=xx[idx].reshape(21,2);clean=x[idx].reshape(21,2)
                for a,b in EDGES:
                    ax.plot(clean[[a,b],0],clean[[a,b],1],color='lightgray');ax.plot(p[[a,b],0],p[[a,b],1],color='tab:blue',lw=1)
                ax.scatter(p[:,0],p[:,1],s=8);ax.set_title(f'{cls} {kind} {level}');ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'{kind}_preview.png',dpi=130);plt.close(fig)
    assert before==snapshot(base)
    save(out/'verification.json',dict(source_unchanged=True,no_test=True,clean_predictions_equal=True,finite_features=True,disjoint_users=True,imputation_train_only=True))
    print(s.to_string(index=False))


if __name__=='__main__':main()
