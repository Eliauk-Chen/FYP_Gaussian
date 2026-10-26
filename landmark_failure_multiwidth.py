"""Clean training, corruption-specific inner-validation selection; fixed 32 centers."""
import argparse
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from landmark_failure_modes import JITTER, COUNTS
from landmark_experiment import CLASSES, ALPHAS, FACTORS
from landmark_robustness import perturb
from multiwidth_experiment import read, save, snapshot, metrics

KINDS={'jitter':JITTER,'missing':COUNTS,'outlier':COUNTS}


def variants(x,kind,seed,mean):
    rng=np.random.default_rng(1000+seed);order=np.argsort(rng.random((len(x),20)),axis=1)+1
    angle=np.random.default_rng(2000+seed).uniform(0,2*np.pi,(len(x),21))
    shifts=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    noise=np.random.default_rng(seed).standard_normal((len(x),21,2));result=[]
    for level in KINDS[kind]:
        if kind=='jitter':xx=perturb(x,noise,level)
        else:
            p=x.reshape(-1,21,2).copy();a=np.arange(len(x))[:,None];b=order[:,:level]
            if kind=='missing':p[a,b]=np.broadcast_to(mean.reshape(1,21,2),p.shape)[a,b]
            else:
                p[a,b]+=shifts[a,b]
                if level:p/=np.sqrt(np.mean(np.sum(p[:,1:]**2,axis=2),axis=1))[:,None,None]
            xx=p.reshape(-1,42)
        result.append(xx)
    return np.array(result)


def val_score(y,pred):
    # pred shape (severity,n); equal weight per severity, then per class.
    scores=[]
    for c in CLASSES:
        tp=((pred==c)&(y[None,:]==c)).sum(axis=1)
        den=(pred==c).sum(axis=1)+(y==c).sum()
        scores.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0))
    perlevel=np.mean(scores,axis=0)
    return float(perlevel.mean()),perlevel.tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_failure_multiwidth_v1'));out=ap.parse_args().out
    out.mkdir(parents=True,exist_ok=False);base=Path('artifacts/hagrid_landmarks_v1');fail=Path('artifacts/hagrid_failure_modes_v1')
    hashes={str(p):snapshot(p) for p in [base,fail]};save(out/'source_hashes.json',hashes)
    save(out/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),kinds=KINDS,alphas=ALPHAS,factors=FACTORS,
        training='Clean inner-training only, no augmentation; existing clean scaler and 32 centers frozen.',
        selection='Separate configuration per corruption; mean severity Macro-F1 including clean on inner validation only. Noise seeds300+fold. Ties retain first.',
        directed='Initialize reselected single width; strict improvement; fixed seed42 node order; max2 rounds;900s per fold/kind.',
        random='10 independent uniform width configurations seeds0..9; same alpha grid; all reported; not compute matched.',
        evaluation='Lock ALL 9 selections before using cached outer corruptions seeds0..9. Prior development users: exploratory, not independent test.',
        output='32 nodes and 32 responses for all RBF models, no width concatenation. No hardware simulation.'))
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(base/'versions.json',out/'versions.json')
    m=pd.read_csv(base/'development_manifest.csv');m.to_csv(out/'manifest.csv',index=False)
    x=np.load(base/'landmark_features.npz',allow_pickle=True)['features'];assert m.split.ne('test').all()
    locked=[];bundles={}
    with threadpool_limits(limits=1):
        for fold in range(3):
            src=base/f'fold_{fold}';ids=read(src/'split_indices.json');a,b,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
            us=[set(m.iloc[v].user_id) for v in [a,b,c]];assert not(us[0]&us[1] or us[0]&us[2] or us[1]&us[2])
            shared=joblib.load(src/'shared.joblib');sc=shared['scaler'];centers=shared['kmeans'].cluster_centers_;pool=shared['sigma_ref']*np.array(FACTORS)
            mean=x[a].mean(axis=0);assert np.allclose(mean,sc.mean_)
            tr=sc.transform(x[a]);yt=m.iloc[a]['class'].to_numpy();yv=m.iloc[b]['class'].to_numpy()
            dt=((tr[:,None]-centers[None])**2).sum(axis=2);rt=np.stack([np.exp(-dt/(2*s*s)) for s in pool])
            for kind in KINDS:
                start=time.monotonic();p=out/f'fold_{fold}'/kind;p.mkdir(parents=True)
                v=variants(x[b],kind,300+fold,mean);np.savez_compressed(p/'validation_features.npz',features=v,sample_ids=m.iloc[b].sample_id.to_numpy(dtype=str))
                va=sc.transform(v.reshape(-1,42));dv=((va[:,None]-centers[None])**2).sum(axis=2);rv=np.stack([np.exp(-dv/(2*s*s)) for s in pool])
                records=[];fits=0
                def ev(name,ft,fv,w=None):
                    nonlocal fits
                    best=None
                    for alpha in ALPHAS:
                        model=RidgeClassifier(alpha=alpha).fit(ft,yt);fits+=1
                        value,levels=val_score(yv,model.predict(fv).reshape(len(KINDS[kind]),len(b)))
                        r=dict(name=name,alpha=alpha,val_objective=value,level_f1=levels,widths=None if w is None else w.tolist());records.append(r)
                        if best is None or value>best['val_objective']:best=dict(**r,estimator=model)
                    return best
                def ew(name,idx):return ev(name,rt[idx,:,np.arange(32)].T,rv[idx,:,np.arange(32)].T,pool[idx])
                models={'A_reselected':ev('A_reselected',tr,va)};best=None;bestidx=None
                for k in range(5):
                    idx=np.full(32,k);r=ew('B_reselected',idx)
                    if best is None or r['val_objective']>best['val_objective']:best=r;bestidx=idx.copy()
                models['B_reselected']=best;current=best.copy();idx=bestidx.copy();moves=[];start_search=time.monotonic();fits_before=fits;expired=False;rng=np.random.default_rng(42)
                for sweep in range(2):
                    changed=False
                    for j in rng.permutation(32):
                        winner=current;winidx=idx.copy()
                        for k in range(5):
                            if k==idx[j]:continue
                            if time.monotonic()-start_search>=900:expired=True;break
                            trial=idx.copy();trial[j]=k;r=ew('C_directed',trial)
                            if r['val_objective']>winner['val_objective']:winner=r;winidx=trial
                        if winner['val_objective']>current['val_objective']:
                            moves.append(dict(sweep=sweep,node=int(j),old_sigma=float(pool[idx[j]]),new_sigma=float(pool[winidx[j]]),before=current['val_objective'],after=winner['val_objective']))
                            current=winner;idx=winidx;changed=True
                        if expired:break
                    if expired or not changed:break
                models['C_directed']=current
                search=dict(moves=moves,configurations=(fits-fits_before)//6,ridge_fits=fits-fits_before,seconds=time.monotonic()-start_search,expired=expired)
                for seed in range(10):models[f'D_random_{seed}']=ew(f'D_random_{seed}',np.random.default_rng(seed).integers(5,size=32))
                for name,r in models.items():locked.append(dict(fold=fold,kind=kind,model=name,**{k:v for k,v in r.items() if k not in ['name','estimator']}))
                joblib.dump(dict(models=models,shared=shared),p/'models.joblib');save(p/'search.json',records);save(p/'coordinate_search.json',search);save(p/'split_indices.json',ids)
                bundles[(fold,kind)]=(shared,models,c)
                print('LOCKED',fold,kind,'seconds',round(time.monotonic()-start,1),'moves',len(moves),flush=True)
        save(out/'locked_selections.json',dict(utc=datetime.now(timezone.utc).isoformat(),selections=locked))
        rows=[];frames=[];classrows=[]
        for seed in range(10):
            cache=np.load(fail/f'corruptions_seed_{seed}.npz');assert np.array_equal(cache['sample_ids'],m.sample_id)
            for key,xx in zip(cache['keys'],cache['features']):
                kind,level=key.split('_');level=float(level);preds={n:np.full(len(x),'',dtype=object) for n in bundles[(0,kind)][1]};seen=np.zeros(len(x),int)
                for fold in range(3):
                    shared,models,idx=bundles[(fold,kind)];seen[idx]+=1;z=shared['scaler'].transform(xx[idx]);d2=((z[:,None]-shared['kmeans'].cluster_centers_[None])**2).sum(axis=2)
                    for name,r in models.items():
                        f=z if r['widths'] is None else np.exp(-d2/(2*np.array(r['widths'])[None]**2))
                        assert np.isfinite(f).all();preds[name][idx]=r['estimator'].predict(f)
                assert (seen==1).all()
                for name,pred in preds.items():
                    met=metrics(m['class'],pred);tag=dict(seed=seed,kind=kind,level=level,model=name)
                    rows.append(dict(**tag,**{k:met[k] for k in ['accuracy','macro_f1','balanced_accuracy']}))
                    for cls in CLASSES:classrows.append(dict(**tag,label=cls,**met['per_class'][cls]))
                    save(out/'metrics'/f'{key}_{seed}_{name}.json',met)
                    f=m[['sample_id','user_id','class']].copy();f['predicted_label']=pred
                    for k,v in tag.items():f[k]=v
                    frames.append(f)
            print('EVALUATED',seed,flush=True)
    pd.DataFrame(locked).to_csv(out/'validation_selections.csv',index=False)
    r=pd.DataFrame(rows);r.to_csv(out/'seed_metrics.csv',index=False);pd.DataFrame(classrows).to_csv(out/'per_class.csv',index=False)
    pd.concat(frames,ignore_index=True).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    s=r.groupby(['kind','level','model']).agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),noise_seed_std=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index();s.to_csv(out/'summary.csv',index=False)
    random=s[s.model.str.startswith('D_')].groupby(['kind','level']).agg(macro_f1_mean=('macro_f1_mean','mean'),width_seed_std=('macro_f1_mean','std')).reset_index();random.to_csv(out/'random_group.csv',index=False)
    s.groupby(['kind','model']).macro_f1_mean.mean().to_csv(out/'severity_average.csv')
    fig,axes=plt.subplots(1,3,figsize=(17,5));old=pd.read_csv(fail/'summary.csv')
    for ax,kind in zip(axes,KINDS):
        for name in ['A_linear','B_rbf32']:
            q=old[(old.kind==kind)&(old.model==name)];ax.plot(q.level,q.macro_f1_mean,'--',label=name+' old')
        for name in ['A_reselected','B_reselected','C_directed']:
            q=s[(s.kind==kind)&(s.model==name)];ax.errorbar(q.level,q.macro_f1_mean,yerr=q.noise_seed_std,marker='o',label=name)
        q=random[random.kind==kind];ax.plot(q.level,q.macro_f1_mean,':',label='Random 10 mean')
        ax.set(title=kind,xlabel='Noise SD / scale' if kind=='jitter' else 'Affected points /20',ylabel='Macro-F1');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert hashes=={p:snapshot(Path(p)) for p in hashes};save(out/'verification.json',dict(sources_unchanged=True,test_evaluated=False,clean_training_only=True,all_configs_locked_before_outer=True))
    print(s[~s.model.str.startswith('D_')].to_string(index=False))


if __name__=='__main__':main()
