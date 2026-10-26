"""Frozen HaGRID single/multiwidth models: paired empirical Cycle evaluation."""
from pathlib import Path
import argparse,json,hashlib,shutil,time
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import confusion_matrix
from hagrid_hardware_multiwidth import transform_from_bundle
from hagrid_hardware_clean import quantize,CLASSES,save
import matplotlib.pyplot as plt

BASE=Path('artifacts/hagrid_hardware_multiwidth_v1')
CACHE=Path('artifacts/hagrid_failure_modes_v1')
from hardware_paths import RESPONSE_BANK
BANK=RESPONSE_BANK
NAMES=['single','directed']

def measures(y,pred):
    cm=confusion_matrix(y,pred,labels=range(4));tp=np.diag(cm);support=cm.sum(1)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(4),where=cm.sum(0)>0)
    recall=tp/support;f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=(precision+recall)>0)
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean())),cm,precision,recall,f1,support

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_hardware_cycle_v1'));out=ap.parse_args().out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    files=[BANK,BASE/'manifest.csv',BASE/'metrics_all_seeds.csv']+list(BASE.glob('fold_*/*/models.joblib'))+list(BASE.glob('fold_*/shared.joblib'))+list(BASE.glob('fold_*/split_indices.json'))+list(CACHE.glob('corruptions_seed_*.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};save(out/'source_hashes.json',hashes);shutil.copy2(__file__,out/Path(__file__).name)
    cycle_seeds=list(range(1000,1010))
    save(out/'protocol.json',dict(models=NAMES,keypoint_seeds=list(range(10)),cycle_seeds=cycle_seeds,
        pairing='Full 10x10 crossed keypoint/Cycle seeds. Per sample one Cycle index shared across all input dimensions/channels, models, corruption severities and keypoint seeds.',
        response='Raw empirical bank values; no adaptive multiplicative noise, no new Gaussian noise, no physical width change.',
        training='None. All saved single/direct models and preprocessing frozen, no selection.',
        baseline='Recompute float64 Cycle mean lookup exactly as prior baseline; require same outer predictions.',
        evaluation='All 786 outer development samples, user-disjoint folds; separate previously selected models per corruption kind. No official test or new independent confirmation.',
        uncertainty='Cycle SD: first average metrics across keypoint seeds per Cycle seed, then sample SD across 10 Cycle seeds. Not population CI; clean is identical across keypoint seeds and is not 100 independent trials.',
        provenance='Stored empirical bank; no new physical measurements. Mean training used all400 cycles, so this is response-sampling sensitivity, not unseen-device or held-out-cycle generalization.'))
    m=pd.read_csv(BASE/'manifest.csv');assert m.split.ne('test').all();m.to_csv(out/'manifest.csv',index=False);y=np.array([CLASSES.index(c) for c in m['class']])
    with np.load(BANK) as z:levels=z['differential_levels'];bank=z['response_samples'];mean=bank.mean(axis=1,dtype=np.float64)
    assert bank.shape==(256,400,16) and np.isfinite(bank).all()
    cycles=np.array([np.random.default_rng(s).integers(0,400,len(m)) for s in cycle_seeds]);np.savez_compressed(out/'cycle_assignments.npz',cycle_seeds=cycle_seeds,indices_zero_based=cycles,sample_ids=m.sample_id.to_numpy(dtype=str))
    folds=[];seen=np.zeros(len(m),int)
    for f in range(3):
        shared=joblib.load(BASE/f'fold_{f}/shared.joblib');ids=json.loads((BASE/f'fold_{f}/split_indices.json').read_text());a,v,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
        users=[set(m.iloc[ix].user_id) for ix in [a,v,c]];assert not(users[0]&users[1] or users[0]&users[2] or users[1]&users[2]);seen[c]+=1
        np.testing.assert_array_equal(shared['levels'],levels);np.testing.assert_array_equal(shared['response_mean'],mean)
        models={}
        for kind in ['jitter','missing','outlier']:
            saved=joblib.load(BASE/f'fold_{f}/{kind}/models.joblib');models[kind]={n:(transform_from_bundle(saved[n]),saved[n]['head']) for n in NAMES}
        folds.append((shared,c,models))
    assert (seen==1).all();rows=[];classrows=[];cms=[];diag=[];baseline_rows=[]
    for ks in range(10):
        with np.load(CACHE/f'corruptions_seed_{ks}.npz') as cache,np.load(BASE/f'predictions_noise_seed_{ks}.npz') as old:
            assert np.array_equal(cache['sample_ids'],m.sample_id);keys=cache['keys'].copy()
            pred=np.empty((10,len(keys),2,len(m)),np.int8);score=np.empty((10,len(keys),2,len(m),4),np.float64)
            for ki,(key,xx) in enumerate(zip(keys,cache['features'])):
                kind,l=key.split('_');level=float(l);bp=np.empty((2,len(m)),np.int8)
                for shared,idx,models in folds:
                    raw=shared['range_scaler'].transform(shared['pca'].transform(shared['scaler'].transform(xx[idx])))
                    q,qi=quantize(np.clip(raw,0,1),levels);hmean=mean[qi].reshape(-1,256)
                    for ni,n in enumerate(NAMES):
                        tr,head=models[kind][n];bp[ni,idx]=head.predict(tr.transform(hmean).astype(float)).argmax(1)
                    for ci,cs in enumerate(cycle_seeds):
                        h=bank[qi,cycles[ci,idx,None]].reshape(-1,256).astype(float)
                        assert h.shape==(len(idx),256)
                        for ni,n in enumerate(NAMES):
                            tr,head=models[kind][n];features=tr.transform(h).astype(float);assert np.isfinite(features).all()
                            ss=head.predict(features);score[ci,ki,ni,idx]=ss;pred[ci,ki,ni,idx]=ss.argmax(1)
                            diag.append(dict(keypoint_seed=ks,cycle_seed=cs,kind=kind,level=level,model=n,zero_elements=int((features==0).sum()),all_zero_samples=int(np.all(features==0,axis=1).sum()),n=len(idx)))
                for ni,n in enumerate(NAMES):
                    oi=list(old['models']).index(n);np.testing.assert_array_equal(bp[ni],old['predicted_labels'][ki,oi])
                    met,*_=measures(y,bp[ni]);baseline_rows.append(dict(keypoint_seed=ks,kind=kind,level=level,model=n,**met))
                for ci,cs in enumerate(cycle_seeds):
                    for ni,n in enumerate(NAMES):
                        met,cm,p,r,f,s=measures(y,pred[ci,ki,ni]);tag=dict(keypoint_seed=ks,cycle_seed=cs,kind=kind,level=level,model=n)
                        rows.append(dict(**tag,**met));cms.append(cm)
                        for j,cl in enumerate(CLASSES):classrows.append(dict(**tag,label=cl,precision=p[j],recall=r[j],f1=f[j],support=int(s[j])))
        np.savez_compressed(out/f'predictions_keypoint_seed_{ks}.npz',cycle_seeds=cycle_seeds,condition_keys=keys,models=NAMES,sample_ids=m.sample_id.to_numpy(dtype=str),classes=CLASSES,true_labels=y,predicted_labels=pred,class_scores=score)
        print('FINISHED KEYPOINT SEED',ks,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'cycle_metrics.csv',index=False);pd.DataFrame(classrows).to_csv(out/'per_class.csv',index=False);pd.DataFrame(diag).to_csv(out/'numerical_diagnostics.csv',index=False)
    cm=np.array(cms);np.savez_compressed(out/'confusions.npz',counts=cm,normalized_true=cm/cm.sum(axis=2,keepdims=True),metadata=df[['keypoint_seed','cycle_seed','kind','level','model']].astype(str).to_numpy(),metadata_columns=['keypoint_seed','cycle_seed','kind','level','model'],classes=CLASSES)
    baseline=pd.DataFrame(baseline_rows);baseline.to_csv(out/'mean_response_metrics.csv',index=False)
    averaged=df.groupby(['kind','level','model','cycle_seed'])[['accuracy','macro_f1','balanced_accuracy']].mean().reset_index();averaged.to_csv(out/'per_cycle_averaged_metrics.csv',index=False)
    summary=averaged.groupby(['kind','level','model']).agg(cycle_accuracy=('accuracy','mean'),cycle_macro_f1=('macro_f1','mean'),cycle_macro_f1_sd=('macro_f1','std'),cycle_balanced_accuracy=('balanced_accuracy','mean')).reset_index()
    means=baseline.groupby(['kind','level','model']).agg(mean_accuracy=('accuracy','mean'),mean_macro_f1=('macro_f1','mean')).reset_index();summary=summary.merge(means,on=['kind','level','model']);summary['cycle_minus_mean_f1']=summary.cycle_macro_f1-summary.mean_macro_f1;summary.to_csv(out/'summary.csv',index=False)
    paired=summary.pivot(index=['kind','level'],columns='model',values=['mean_macro_f1','cycle_macro_f1']);paired.columns=['_'.join(c) for c in paired.columns];paired=paired.reset_index();paired['mean_multi_gain']=paired.mean_macro_f1_directed-paired.mean_macro_f1_single;paired['cycle_multi_gain']=paired.cycle_macro_f1_directed-paired.cycle_macro_f1_single;paired.to_csv(out/'paired_gains.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,kind in zip(axes,['jitter','missing','outlier']):
        for name,color in [('single','tab:blue'),('directed','tab:orange')]:
            g=summary[(summary.kind==kind)&(summary.model==name)];ax.plot(g.level,g.mean_macro_f1,'--',color=color,label=name+' mean');ax.errorbar(g.level,g.cycle_macro_f1,yerr=g.cycle_macro_f1_sd,marker='o',color=color,label=name+' Cycle')
        ax.set(title=kind,xlabel='Noise SD / RMS' if kind=='jitter' else 'Affected points /20',ylabel='Macro-F1',ylim=(0,1.03));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cycle_robustness.png',dpi=180);plt.close(fig)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out/'verification.json',dict(sources_unchanged=True,mean_predictions_exact=True,finite_features=True,user_disjoint=True,refitted=False,official_test_used=False,prediction_count=10*10*23*2*len(m),elapsed_seconds=time.monotonic()-start))
    print(summary[summary.level.isin([0,.2,.7,14])].to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
