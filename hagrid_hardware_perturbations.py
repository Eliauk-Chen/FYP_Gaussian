"""Replay the exact old corruption cache through frozen hardware-aligned models."""
import csv, gzip, hashlib, json, shutil, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from hagrid_hardware_clean import quantize, scores, save, CLASSES, FullJointGaussianTransformer
import matplotlib.pyplot as plt

BASE=Path('artifacts/hagrid_hardware_clean_v2')
CACHE=Path('artifacts/hagrid_failure_modes_v1')
OUT=Path('artifacts/hagrid_hardware_perturbations_v1')
MODELS=['linear16','ideal256_matched','hardware256']

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    paths=list(BASE.rglob('*'));paths=[p for p in paths if p.is_file()]
    paths+=list(CACHE.glob('corruptions_seed_*.npz'))+[CACHE/'protocol.json',Path('artifacts/hagrid_landmarks_v1/landmark_features.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(OUT/'source_hashes.json',hashes)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    old=json.loads((CACHE/'protocol.json').read_text())
    save(OUT/'protocol.json',dict(original_corruption_protocol=old,models=MODELS,
         evaluation='Exact cached corruptions, seeds 0..9; reuse frozen clean v2 models. No fitting, no selection, no official test. Pooled outer predictions, all 786 samples once per condition.',
         hardware='400-Cycle mean lookup, no cycle sampling or added noise',
         uncertainty='Sample SD across 10 corruption seeds, not population confidence or 10 independent datasets',
         warnings=['Missing-point imputation assumes clean normalization scale is available','PCA and input clipping may alter perturbation effect; record clipping','Old development users reused: exploratory only']))
    m=pd.read_csv(BASE/'manifest.csv');assert m.split.ne('test').all()
    m.to_csv(OUT/'manifest.csv',index=False)
    x=np.load('artifacts/hagrid_landmarks_v1/landmark_features.npz',allow_pickle=True)['features']
    y=np.array([CLASSES.index(c) for c in m['class']]);folds=[];seen=np.zeros(len(x),int)
    fill=np.zeros_like(x);foldof=np.full(len(x),-1)
    for fold in range(3):
        b=joblib.load(BASE/f'fold_{fold}/model_bundle.joblib');ids=json.loads((BASE/f'fold_{fold}/split_indices.json').read_text())
        a,v,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
        us=[set(m.iloc[i].user_id) for i in [a,v,c]];assert not(us[0]&us[1] or us[0]&us[2] or us[1]&us[2])
        seen[c]+=1;fill[c]=x[a].mean(axis=0);foldof[c]=fold
        folds.append((b,FullJointGaussianTransformer.from_state_dict(b['transform_state']),c))
    assert (seen==1).all()
    original=pd.read_csv(BASE/'outer_predictions.csv')
    expected={n:original[original.model==n].set_index('sample_id').loc[m.sample_id].predicted_label.to_numpy() for n in MODELS}
    rows=[];perclass=[];diagnostics=[];confusions=[]
    with gzip.open(OUT/'predictions.csv.gz','wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seed','kind','level','model','fold','sample_id','user_id','true_label','predicted_label']+[f'score_{c}' for c in CLASSES])
        for seed in range(10):
            with np.load(CACHE/f'corruptions_seed_{seed}.npz') as z:
                assert np.array_equal(z['sample_ids'],m.sample_id) and np.array_equal(z['fold'],foldof)
                for key,xx in zip(z['keys'],z['features']):
                    kind,ls=key.split('_');level=float(ls)
                    assert np.isfinite(xx).all()
                    # Independently reconstruct each cached perturbation from its saved random draws.
                    p=x.reshape(-1,21,2).copy();sample=np.arange(len(x))[:,None]
                    if kind=='jitter' and level:
                        p+=level*z['unit_noise'];p-=p[:,:1,:]
                        p/=np.sqrt(np.mean(np.sum(p[:,1:]**2,axis=2),axis=1))[:,None,None]
                    elif kind in ['missing','outlier'] and level:
                        indices=z['point_order'][:,:int(level)]
                        if kind=='missing':p[sample,indices]=fill.reshape(-1,21,2)[sample,indices]
                        else:
                            shifts=np.stack([np.cos(z['angles']),np.sin(z['angles'])],axis=-1)
                            p[sample,indices]+=shifts[sample,indices]
                            p/=np.sqrt(np.mean(np.sum(p[:,1:]**2,axis=2),axis=1))[:,None,None]
                    np.testing.assert_allclose(xx,p.reshape(-1,42),atol=1e-14,rtol=1e-14)
                    pred={n:np.full(len(x),-1) for n in MODELS}
                    for fold,(b,tr,idx) in enumerate(folds):
                        raw=b['range_scaler'].transform(b['pca'].transform(b['scaler'].transform(xx[idx])))
                        q,qi=quantize(np.clip(raw,0,1),b['levels'])
                        h=b['response_mean'][qi].reshape(-1,256);joint=tr.transform(h)
                        assert np.isfinite(joint).all()
                        diagnostics.append(dict(seed=seed,kind=kind,level=level,fold=fold,n=len(idx),clipped_values=int(((raw<0)|(raw>1)).sum()),input_values=raw.size,joint_zero_fraction=float((joint==0).mean()),all_zero_rows=int(np.all(joint==0,axis=1).sum())))
                        for name,head,features in [('linear16',b['linear_head'],q),('ideal256_matched',b['ideal_head'],tr.ideal_pc_transform(q)),('hardware256',b['hardware_head'],joint)]:
                            ss=head.predict(features);yp=ss.argmax(axis=1);pred[name][idx]=yp
                            for j,i in enumerate(idx):writer.writerow([seed,kind,level,name,fold,m.iloc[i].sample_id,m.iloc[i].user_id,CLASSES[y[i]],CLASSES[yp[j]],*ss[j].tolist()])
                    for name,yp in pred.items():
                        if level==0:assert np.array_equal(np.array(CLASSES)[yp],expected[name])
                        tag=dict(seed=seed,kind=kind,level=level,model=name)
                        rows.append(dict(**tag,**scores(y,yp)))
                        report=classification_report(y,yp,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0)
                        for cls in CLASSES:perclass.append(dict(**tag,label=cls,**report[cls]))
                        confusions.append(dict(**tag,counts=confusion_matrix(y,yp,labels=range(4)).tolist(),normalized_true=confusion_matrix(y,yp,labels=range(4),normalize='true').tolist()))
            print('FINISHED SEED',seed,flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/'seed_metrics.csv',index=False)
    s=r.groupby(['kind','level','model']).agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index()
    s.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(perclass).to_csv(OUT/'per_class.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'response_diagnostics.csv',index=False)
    save(OUT/'confusions.json',confusions)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,kind in zip(axes,['jitter','missing','outlier']):
        for name in MODELS:
            a=s[(s.kind==kind)&(s.model==name)]
            ax.errorbar(a.level*(100 if kind=='jitter' else 1),a.macro_f1_mean,yerr=a.macro_f1_sd,marker='o',label=name,capsize=2)
        ax.set(title=kind,xlabel='Noise SD / clean RMS (%)' if kind=='jitter' else 'Affected non-wrist points /20',ylabel='Macro-F1 (10 seeds)',ylim=(0,1.03));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'perturbation_curves.png',dpi=180);plt.close(fig)
    for kind in ['jitter','missing','outlier']:shutil.copy2(CACHE/f'{kind}_preview.png',OUT/f'{kind}_preview.png')
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(OUT/'verification.json',dict(sources_unchanged=True,cache_reconstruction_verified=True,clean_predictions_match=True,user_disjoint=True,models_refitted=False,test_evaluated=False,seeds=10,conditions_per_seed=23,models=3,prediction_rows=len(x)*10*23*3,elapsed_seconds=time.perf_counter()-start))
    print(s[s.level.isin([.2,.5,.7,4,14])].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
