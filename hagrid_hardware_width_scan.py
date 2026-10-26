"""Validation-only shared target-width selection; no change to empirical curves."""
from pathlib import Path
import json,hashlib,shutil,time,gzip,csv
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
from hagrid_hardware_clean import quantize,scores,save,CLASSES,ALPHAS,FullJointGaussianTransformer
from landmark_failure_multiwidth import variants,KINDS
import matplotlib.pyplot as plt

BASE=Path('artifacts/hagrid_hardware_clean_v2')
CACHE=Path('artifacts/hagrid_failure_modes_v1')
OUT=Path('artifacts/hagrid_hardware_width_scan_v1')
POOL=[.125,.175,.25,.3,.35,.5,.7,1.0]

def response(b,x):
    q,idx=quantize(np.clip(b['range_scaler'].transform(b['pca'].transform(b['scaler'].transform(x))),0,1),b['levels'])
    return q,b['response_mean'][idx].reshape(-1,256)

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    files=[p for p in BASE.rglob('*') if p.is_file()]+list(CACHE.glob('corruptions_seed_*.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};save(OUT/'source_hashes.json',hashes)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    save(OUT/'protocol.json',dict(widths=POOL,alphas=ALPHAS,validation_seeds=[300,301,302],conditions=KINDS,
        objective='Separate selection per corruption, equal mean Macro-F1 across all severities including clean; ties retain first',
        fit='Clean inner training only, no augmentation; existing PCA/scalers/quantization/256 centers fixed; calibration alpha and factor bounds fixed per fold. Calibration weights necessarily refit for each target width.',
        controls=['original frozen Gaussian','fixed original width, alpha reselected on perturbed validation','width and alpha selected on same perturbed validation'],
        hardware='Unchanged empirical mean-response bank, no extra noise. Width is digital ideal calibration target, not measured device width.',
        evaluation='All 9 configs locked before outer cache evaluation. Seeds0..9, old user-disjoint development folds, no official test. Exploratory, no new independent confirmation.',
        scope='Shared scalar width only, no per-node multiwidth or Linear comparison'))
    m=pd.read_csv(BASE/'manifest.csv');m.to_csv(OUT/'manifest.csv',index=False)
    x=np.load('artifacts/hagrid_landmarks_v1/landmark_features.npz',allow_pickle=True)['features']
    y=np.array([CLASSES.index(c) for c in m['class']]);assert m.split.ne('test').all()
    records=[];locks=[];bundles=[];seen=np.zeros(len(x),int)
    for fold in range(3):
        b=joblib.load(BASE/f'fold_{fold}/model_bundle.joblib');ids=json.loads((BASE/f'fold_{fold}/split_indices.json').read_text())
        a,v,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']];seen[c]+=1
        users=[set(m.iloc[ix].user_id) for ix in [a,v,c]];assert not(users[0]&users[1] or users[0]&users[2] or users[1]&users[2])
        qa,ha=response(b,x[a]);original=FullJointGaussianTransformer.from_state_dict(b['transform_state'])
        candidates={}
        for sigma in POOL:
            tr=FullJointGaussianTransformer(16,16,256,sigma,original.calibration_alpha,original.factor_lower,original.factor_upper,original.epsilon,42,300).fit(ha,qa,preset_centers=original.centers_)
            np.testing.assert_allclose(tr.channel_mean_,original.channel_mean_)
            np.testing.assert_allclose(tr.channel_std_,original.channel_std_)
            features=tr.transform(ha).astype(float)
            heads=[Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(features,np.eye(4)[y[a]]) for alpha in ALPHAS]
            candidates[sigma]=(tr,heads)
        foldmodels={}
        for kind in KINDS:
            vv=variants(x[v],kind,300+fold,x[a].mean(axis=0))
            _,hv=response(b,vv.reshape(-1,42));best=None;control=None
            for sigma,(tr,heads) in candidates.items():
                fv=tr.transform(hv).astype(float)
                for alpha,head in zip(ALPHAS,heads):
                    pred=head.predict(fv).argmax(axis=1).reshape(len(KINDS[kind]),len(v))
                    vals=[f1_score(y[v],pp,labels=range(4),average='macro',zero_division=0) for pp in pred]
                    rec=dict(fold=fold,kind=kind,sigma=sigma,alpha=alpha,val_objective=float(np.mean(vals)),per_level_f1=vals)
                    records.append(rec)
                    if best is None or rec['val_objective']>best[0]['val_objective']:best=(rec,tr,head)
                    if sigma==original.joint_sigma and (control is None or rec['val_objective']>control[0]['val_objective']):control=(rec,tr,head)
            assert control is not None
            models={'fixed_width_alpha_tuned':control,'width_tuned':best};foldmodels[kind]=models
            locks.append(dict(fold=fold,kind=kind,original_sigma=original.joint_sigma,original_alpha=b['config']['alpha'],fixed_width_alpha_tuned=control[0],width_tuned=best[0]))
            print('LOCKED',fold,kind,'sigma',best[0]['sigma'],'val',best[0]['val_objective'],flush=True)
        folder=OUT/f'fold_{fold}';folder.mkdir()
        joblib.dump(dict(shared=b,models={kind:{name:dict(config=r,transform_state=tr.state_dict(),head=head) for name,(r,tr,head) in models.items()} for kind,models in foldmodels.items()}),folder/'models.joblib')
        save(folder/'split_indices.json',ids);bundles.append((b,c,foldmodels))
    assert (seen==1).all()
    save(OUT/'locked_configs.json',locks);save(OUT/'validation_search.json',records)
    rows=[];details=[]
    samples=m.sample_id.to_numpy();users=m.user_id.to_numpy()
    with gzip.open(OUT/'predictions.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','kind','level','model','sample_id','user_id','true_label','predicted_label']+[f'score_{c}' for c in CLASSES])
        for seed in range(10):
            with np.load(CACHE/f'corruptions_seed_{seed}.npz') as z:
                assert np.array_equal(z['sample_ids'],samples)
                for key,xx in zip(z['keys'],z['features']):
                    kind,level=key.split('_');level=float(level);preds={n:np.full(len(x),-1) for n in ['fixed_width_alpha_tuned','width_tuned']}
                    for b,idx,models in bundles:
                        _,h=response(b,xx[idx])
                        for name,(_,tr,head) in models[kind].items():
                            ss=head.predict(tr.transform(h).astype(float));yp=ss.argmax(axis=1);preds[name][idx]=yp
                            w.writerows([seed,kind,level,name,samples[i],users[i],CLASSES[y[i]],CLASSES[yp[j]],*ss[j].tolist()] for j,i in enumerate(idx))
                    for name,yp in preds.items():
                        tag=dict(seed=seed,kind=kind,level=level,model=name);rows.append(dict(**tag,**scores(y,yp)))
                        details.append(dict(**tag,per_class=classification_report(y,yp,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0),confusion_counts=confusion_matrix(y,yp,labels=range(4)).tolist(),confusion_normalized=confusion_matrix(y,yp,labels=range(4),normalize='true').tolist()))
            print('EVALUATED',seed,flush=True)
    old=pd.read_csv('artifacts/hagrid_hardware_perturbations_v1/seed_metrics.csv');old=old[old.model=='hardware256'].copy();old['model']='original_frozen'
    r=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);r.to_csv(OUT/'seed_metrics.csv',index=False)
    s=r.groupby(['kind','level','model']).agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index();s.to_csv(OUT/'summary.csv',index=False)
    save(OUT/'detailed_metrics.json',details)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,kind in zip(axes,KINDS):
        for name,g in s[s.kind==kind].groupby('model'):ax.errorbar(g.level,g.macro_f1_mean,yerr=g.macro_f1_sd,marker='o',label=name)
        ax.set(title=kind,xlabel='SD / RMS' if kind=='jitter' else 'Affected points /20',ylabel='Macro-F1',ylim=(0,1.03));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'outer_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));df=pd.DataFrame(records)
    for ax,kind in zip(axes,KINDS):
        for fold in range(3):
            g=df[(df.kind==kind)&(df.fold==fold)].groupby('sigma').val_objective.max();ax.plot(g.index,g.values,'o-',label=f'fold {fold}')
        ax.set(title=kind,xlabel='Shared target sigma',ylabel='Validation mean severity Macro-F1');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'validation_width_scan.png',dpi=170);plt.close(fig)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(OUT/'verification.json',dict(sources_unchanged=True,test_evaluated=False,width_candidates=8,validation_fits=3*8*len(ALPHAS),candidate_evaluations=len(records),elapsed_seconds=time.monotonic()-start))
    print(s[s.level.isin([0,.2,.7,14])].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
