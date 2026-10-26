"""Frozen clean models under scale-relative landmark jitter; development only."""
import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits
from landmark_experiment import CLASSES, ALPHAS, FACTORS, EDGES
from multiwidth_experiment import read, save, snapshot, metrics, score, phi

LEVELS=[0., .01, .025, .05, .1, .2]
SEEDS=list(range(10))
NAMES=['A_linear','B_rbf32','C_multiwidth32','K_full_gaussian']


def perturb(x, noise, level):
    """Equivalent to pixel noise SD=level*clean RMS, before normalization."""
    if level==0:
        return x.copy()
    p=x.reshape(-1,21,2)+level*noise
    p=p-p[:,:1,:]
    rms=np.sqrt(np.mean(np.sum(p[:,1:]**2,axis=2),axis=1))
    assert np.isfinite(rms).all() and (rms>1e-8).all()
    return (p/rms[:,None,None]).reshape(-1,42)


def fit_multi(base, out, x, m):
    """All selection uses clean inner validation, never corrupted outcomes."""
    splits=read(base/'split_indices.json')
    shared=joblib.load(base/'shared.joblib')
    a,b,c=[np.asarray(splits[k]) for k in ['train','val','outer_holdout']]
    users=[set(m.iloc[v].user_id) for v in [a,b,c]]
    assert not (users[0]&users[1] or users[0]&users[2] or users[1]&users[2])
    sc=shared['scaler']; centers=shared['kmeans'].cluster_centers_
    assert np.allclose(sc.mean_,x[a].mean(axis=0))
    tr,va=sc.transform(x[a]),sc.transform(x[b])
    ytr,yval=m.iloc[a]['class'],m.iloc[b]['class']
    single=joblib.load(base/'B_rbf32.joblib')
    widths=np.full(len(centers),single['config']['sigma'])
    pool=np.array(FACTORS)*shared['sigma_ref']
    model=single['model'];best=score(yval,model.predict(phi(va,centers,widths)))
    alpha=single['config']['alpha'];records=[];moves=[];fits=0
    start=time.monotonic();rng=np.random.default_rng(42);expired=False
    for sweep in range(2):
        changed=False
        for j in rng.permutation(len(centers)):
            old=float(widths[j]);best_node=None
            for candidate in pool:
                if np.isclose(candidate,old):continue
                if time.monotonic()-start>=900:expired=True;break
                w=widths.copy();w[j]=candidate
                ft,fv=phi(tr,centers,w),phi(va,centers,w)
                for al in ALPHAS:
                    fitted=RidgeClassifier(alpha=al).fit(ft,ytr);fits+=1
                    f=score(yval,fitted.predict(fv))
                    records.append(dict(sweep=sweep,node=int(j),candidate_sigma=float(candidate),alpha=al,val_macro_f1=f,widths=w.tolist()))
                    if f>best and (best_node is None or f>best_node[0]):
                        best_node=(f,w.copy(),fitted,al)
            if best_node is not None:
                new,w,model,alpha=best_node
                moves.append(dict(sweep=sweep,node=int(j),old_sigma=old,new_sigma=float(w[j]),old_f1=best,new_f1=new))
                best=new;widths=w;changed=True
            if expired:break
        if expired or not changed:break
    out.mkdir()
    bundle=dict(model=model,widths=widths,alpha=alpha,clean_val_macro_f1=best)
    joblib.dump(bundle,out/'C_multiwidth32.joblib')
    save(out/'search.json',records)
    save(out/'selection.json',dict(widths=widths.tolist(),alpha=alpha,val_macro_f1=best,accepted_moves=moves,
        ridge_fits=fits,candidate_configurations=fits//len(ALPHAS),seconds=time.monotonic()-start,budget_expired=expired))
    save(out/'split_indices.json',splits)
    return shared,bundle,c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_landmark_jitter_v1'))
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    base=Path('artifacts/hagrid_landmarks_v1');before=snapshot(base)
    save(out/'source_hashes.json',before)
    save(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),levels=LEVELS,seeds=SEEDS,models=NAMES,
        perturbation='Independent normal x/y offsets for all 21 points including wrist, SD=level*clean RMS hand scale; recenter at noisy wrist and recompute RMS; no clipping.',
        common_random_numbers='Same noise direction per sample/seed across severity and models; PCG64 default_rng(seed).',
        selection='Existing clean models reused; multiwidth clean inner-validation coordinate search only, seed42, max2 sweeps,900s/fold, strict improvements.',
        evaluation='786 development samples, prior nested outer folds. No test data; no augmentation or retraining on noise.',
        limitation='Hypothetical keypoint localization noise, not image blur, missing points, or measured device noise.'))
    m=pd.read_csv(base/'development_manifest.csv');cache=np.load(base/'landmark_features.npz',allow_pickle=True)
    x=cache['features'];assert np.array_equal(cache['sample_ids'],m.sample_id)
    assert m.user_id.is_unique and m.split.ne('test').all()
    m.to_csv(out/'manifest.csv',index=False)
    shutil.copy2(base/'versions.json',out/'versions.json')
    shutil.copy2(__file__,out/Path(__file__).name)
    bundles=[]
    with threadpool_limits(limits=1):
        # Complete and lock ALL clean selections before any corruption evaluation.
        for fold in range(3):
            shared,multi,idx=fit_multi(base/f'fold_{fold}',out/f'fold_{fold}',x,m)
            models={n:joblib.load(base/f'fold_{fold}'/(n+'.joblib')) for n in NAMES if n!='C_multiwidth32'}
            bundles.append((shared,multi,idx,models))
            print('CLEAN CONFIG LOCKED',fold,flush=True)
        save(out/'all_configs_locked.json',dict(utc=datetime.now(timezone.utc).isoformat(),folds=3))
        all_rows=[];summaries=[];perclass=[];foldmetrics=[];noise_audit=[]
        for seed in SEEDS:
            noise=np.random.default_rng(seed).standard_normal((len(x),21,2))
            variants=np.array([perturb(x,noise,t) for t in LEVELS])
            np.savez_compressed(out/f'perturbed_seed_{seed}.npz',features=variants,levels=LEVELS,unit_noise=noise,sample_ids=m.sample_id.to_numpy(dtype=str))
            for level,xx in zip(LEVELS,variants):
                assert np.isfinite(xx).all() and np.allclose(xx[:,:2],0)
                assert np.allclose(np.mean(np.sum(xx.reshape(-1,21,2)[:,1:]**2,axis=2),axis=1),1)
                noise_audit.append(dict(seed=seed,level=level,normalized_feature_rmse=float(np.sqrt(np.mean((xx-x)**2)))))
                preds={n:np.full(len(x),'',dtype=object) for n in NAMES};seen=np.zeros(len(x),dtype=int)
                for fold,(shared,multi,idx,models) in enumerate(bundles):
                    seen[idx]+=1;z=shared['scaler'].transform(xx[idx]);centers=shared['kmeans'].cluster_centers_
                    for name in NAMES:
                        if name=='C_multiwidth32':pred=multi['model'].predict(phi(z,centers,multi['widths']))
                        elif name=='A_linear':pred=models[name]['model'].predict(z)
                        elif name=='B_rbf32':
                            b=models[name];pred=b['model'].predict(phi(z,centers,[b['config']['sigma']]*len(centers)))
                        else:
                            b=models[name];e=b['extra'];k=e['centerer'].transform(rbf_kernel(z,e['train_features'],gamma=e['gamma']))
                            pred=np.array(CLASSES)[(b['model'].predict(k)+e['target_mean']).argmax(axis=1)]
                        if level==0 and name!='C_multiwidth32':
                            old=pd.read_csv(base/f'fold_{fold}'/(name+'_outer_predictions.csv'))
                            assert np.array_equal(old.sample_id,m.iloc[idx].sample_id) and np.array_equal(old.predicted_label,pred)
                        preds[name][idx]=pred
                        met=metrics(m.iloc[idx]['class'],pred)
                        foldmetrics.append(dict(seed=seed,level=level,fold=fold,model=name,**{k:met[k] for k in ['accuracy','macro_f1','balanced_accuracy']}))
                assert (seen==1).all()
                for name,pred in preds.items():
                    met=metrics(m['class'],pred)
                    summaries.append(dict(seed=seed,level=level,model=name,**{k:met[k] for k in ['accuracy','macro_f1','balanced_accuracy']}))
                    for cls in CLASSES:perclass.append(dict(seed=seed,level=level,model=name,label=cls,**met['per_class'][cls]))
                    save(out/'metrics'/f'{name}_level{level}_seed{seed}.json',met)
                    frame=m[['sample_id','user_id','class']].copy();frame['predicted_label']=pred
                    frame['seed']=seed;frame['level']=level;frame['model']=name;all_rows.append(frame)
            print('EVALUATED NOISE SEED',seed,flush=True)
    pd.concat(all_rows,ignore_index=True).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(foldmetrics).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'per_class.csv',index=False)
    pd.DataFrame(noise_audit).to_csv(out/'noise_audit.csv',index=False)
    results=pd.DataFrame(summaries);results.to_csv(out/'seed_metrics.csv',index=False)
    summary=results.groupby(['model','level']).agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),macro_f1_seed_std=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index()
    summary['f1_drop_from_clean']=summary.groupby('model').macro_f1_mean.transform('first')-summary.macro_f1_mean
    summary.to_csv(out/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,r in summary.groupby('model'):
        ax.errorbar(100*r.level,r.macro_f1_mean,yerr=r.macro_f1_seed_std,label=name,marker='o',capsize=3)
    ax.set(xlabel='Per-coordinate noise SD / clean hand RMS (%)',ylabel='Pooled outer-fold Macro-F1',title='Clean-trained frozen models; jitter seeds 0-9')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'robustness_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,len(LEVELS),figsize=(15,10))
    noise=np.random.default_rng(0).standard_normal((len(x),21,2))
    for row,cls in enumerate(CLASSES):
        i=int(np.flatnonzero(m['class'].eq(cls))[0]);original=x[i].reshape(21,2)
        for col,t in enumerate(LEVELS):
            ax=axes[row,col];p=original+t*noise[i]
            for a,b in EDGES:
                ax.plot(original[[a,b],0],original[[a,b],1],color='lightgray',lw=2)
                ax.plot(p[[a,b],0],p[[a,b],1],color='tab:blue',lw=1)
            ax.scatter(p[:,0],p[:,1],s=8);ax.set_title(f'{cls}: {100*t:g}%');ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
    fig.suptitle('First sample per class, seed 0: clean gray / perturbed blue (before renormalization)')
    fig.tight_layout();fig.savefig(out/'jitter_preview.png',dpi=140);plt.close(fig)
    unchanged=before==snapshot(base);assert unchanged
    save(out/'verification.json',dict(source_unchanged=unchanged,clean_predictions_match=True,user_disjoint=True,outer_coverage_once=True,test_evaluated=False,finite_features=True))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
