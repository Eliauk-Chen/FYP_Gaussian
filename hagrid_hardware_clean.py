"""Clean-input development baseline using the original empirical hardware transform."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys, json, hashlib, shutil, time
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hardware_paths import HARDWARE_ROOT
ROOT=HARDWARE_ROOT
sys.path.insert(0,str(ROOT/'src'))
from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer, benchmark_kmeans
BASE=Path('artifacts/hagrid_landmarks_v1')
OUT=Path('artifacts/hagrid_hardware_clean_v2')
CLASSES=['fist','like','peace','ok']
ALPHAS=[1e-8,1e-6,1e-4,.001,.01,.1,1,10,100]

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def quantize(x,levels):
    # Same nearest-level rule as original quantizer, including upper tie break.
    upper=np.clip(np.searchsorted(levels,x),0,len(levels)-1)
    lower=np.maximum(upper-1,0)
    idx=np.where(abs(x-levels[lower])<abs(x-levels[upper]),lower,upper)
    return levels[idx],idx

def scores(y,p):
    return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro'),balanced_accuracy=balanced_accuracy_score(y,p))

def choose(a,b,y,yv,records,tag):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    best=None
    for alpha in ALPHAS:
        model=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(a,np.eye(4)[y])
        pred=model.predict(b).argmax(axis=1)
        r=dict(**tag,alpha=alpha,**scores(yv,pred));records.append(r)
        if best is None or r['macro_f1']>best[0]['macro_f1']:
            best=(r,model)
    return best

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    torch.set_num_threads(1)
    source_files=[BASE/'development_manifest.csv',BASE/'landmark_features.npz',ROOT/'dataset/empirical/hardware_response_samples.npz',ROOT/'src/rbf_hardware/modeling/joint_gaussian.py']
    source_files += [BASE/f'fold_{i}/split_indices.json' for i in range(3)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    save(OUT/'source_hashes.json',hashes)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    save(OUT/'protocol.json',dict(input='Official automatic landmarks, wrist/RMS normalized 42D; no added corruption',
        preprocessing='Train-only StandardScaler -> PCA16 -> train-min/max scaling -> clip [0,1] -> nearest bank level',
        empirical='Cycle mean, no extra noise, no cycle sampling; not ideal noiseless device or live measurement',
        evaluation='Reuse all 3 nested user-disjoint development folds. Lock every configuration before outer predictions. No official test loaded. Exploratory only.',
        models=['linear16','ideal256_matched','hardware256'],
        selection='Validation Macro-F1, first tie retained. Matched ideal uses hardware-selected sigma and head alpha.',
        sigmas=[.25,.3,.35],calibration_alphas=[.01,.1,1],factor_uppers=[1,1.2],head_alphas=ALPHAS,
        centers=256,seed=42,center_algorithm='Original benchmark_kmeans',factor_lower=1e-6,epsilon=1e-12,
        classifier='One-hot multioutput Ridge, unpenalized intercept; same objective as original RidgeLinearClassifier, sklearn float64 solver',
        scope='Single shared width only; no hardware or keypoint perturbation; no augmentation'))
    m=pd.read_csv(BASE/'development_manifest.csv')
    with np.load(BASE/'landmark_features.npz',allow_pickle=True) as z:
        x=z['features'];assert np.array_equal(z['sample_ids'],m.sample_id)
    assert x.shape==(len(m),42) and m.split.ne('test').all() and np.isfinite(x).all()
    with np.load(ROOT/'dataset/empirical/hardware_response_samples.npz') as z:
        levels=z['differential_levels'];mean=z['response_samples'].mean(axis=1,dtype=np.float64)
    y=np.array([CLASSES.index(c) for c in m['class']])
    m.to_csv(OUT/'manifest.csv',index=False)
    bundles=[];records=[];audit=[];locks=[]
    for fold in range(3):
        folder=OUT/f'fold_{fold}';folder.mkdir()
        ids=json.loads((BASE/f'fold_{fold}/split_indices.json').read_text())
        a,b,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
        users=[set(m.iloc[i].user_id) for i in [a,b,c]]
        assert not(users[0]&users[1] or users[0]&users[2] or users[1]&users[2])
        save(folder/'split_indices.json',ids)
        for name,idx in zip(['train','val','outer_holdout'],[a,b,c]):
            audit.append(dict(fold=fold,split=name,n=len(idx),users=m.iloc[idx].user_id.nunique(),classes=m.iloc[idx]['class'].value_counts().to_dict(),user_intersections=0))
        sc=StandardScaler().fit(x[a]);pca=PCA(n_components=16,svd_solver='full').fit(sc.transform(x[a]))
        mm=MinMaxScaler().fit(pca.transform(sc.transform(x[a])))
        def prepare(xx):
            raw=mm.transform(pca.transform(sc.transform(xx)))
            q,idx=quantize(np.clip(raw,0,1),levels)
            return q,mean[idx].reshape(-1,256),float(np.mean((raw<0)|(raw>1)))
        qa,ha,_=prepare(x[a]);qb,hb,clipb=prepare(x[b])
        centers=benchmark_kmeans(qa,centers=256,random_state=42,max_iter=300)
        linear=choose(qa,qb,y[a],y[b],records,dict(fold=fold,model='linear16'))
        best=None
        for sigma in [.25,.3,.35]:
            for cal in [.01,.1,1]:
                for upper in [1.,1.2]:
                    tr=FullJointGaussianTransformer(16,16,256,sigma,cal,1e-6,upper,1e-12,42,300).fit(ha,qa,preset_centers=centers)
                    fa,fb=tr.transform(ha),tr.transform(hb)
                    result=choose(fa,fb,y[a],y[b],records,dict(fold=fold,model='hardware256',sigma=sigma,calibration_alpha=cal,factor_upper=upper))
                    if best is None or result[0]['macro_f1']>best[0]['macro_f1']:
                        best=(*result,tr)
        r,head,tr=best
        ideal=Ridge(alpha=r['alpha'],fit_intercept=True,solver='cholesky').fit(tr.ideal_pc_transform(qa).astype(float),np.eye(4)[y[a]])
        bundle=dict(scaler=sc,pca=pca,range_scaler=mm,levels=levels,response_mean=mean,
                    transform_state=tr.state_dict(),linear_head=linear[1],hardware_head=head,ideal_head=ideal,classes=CLASSES,
                    config=r,linear_config=linear[0])
        joblib.dump(bundle,folder/'model_bundle.joblib')
        np.savez_compressed(folder/'train_val_cache.npz',train_input=qa,val_input=qb,train_hardware=ha,val_hardware=hb,
                            train_joint=tr.transform(ha),val_joint=tr.transform(hb),train_ids=m.iloc[a].sample_id.to_numpy(dtype=str),val_ids=m.iloc[b].sample_id.to_numpy(dtype=str))
        locks.append(dict(fold=fold,hardware=r,linear=linear[0],ideal_validation=scores(y[b],ideal.predict(tr.ideal_pc_transform(qb)).argmax(axis=1)),
                          val_input_clipped_fraction=clipb,pca_explained_variance=float(pca.explained_variance_ratio_.sum())))
        bundles.append((bundle,c))
        print('LOCKED',fold,r,flush=True)
    save(OUT/'locked_configs.json',locks)
    save(OUT/'data_audit.json',audit)
    pd.DataFrame(records).to_csv(OUT/'validation_search.csv',index=False)
    # All validation decisions are persisted before any outer prediction/metric.
    predictions=[];diagnostics=[];pooled={k:np.full(len(x),-1) for k in ['linear16','ideal256_matched','hardware256']}
    seen=np.zeros(len(x),int)
    for fold,(bundle,c) in enumerate(bundles):
        sc,pca,mm=[bundle[k] for k in ['scaler','pca','range_scaler']]
        raw=mm.transform(pca.transform(sc.transform(x[c])))
        q,idx=quantize(np.clip(raw,0,1),levels);h=mean[idx].reshape(-1,256)
        tr=FullJointGaussianTransformer.from_state_dict(bundle['transform_state'])
        f=tr.transform(h);ideal_f=tr.ideal_pc_transform(q)
        diagnostics.append(dict(fold=fold,clipped_fraction=float(np.mean((raw<0)|(raw>1))),zero_joint_fraction=float(np.mean(f==0)),joint_min=float(f.min()),joint_max=float(f.max())))
        np.savez_compressed(OUT/f'fold_{fold}/outer_cache.npz',inputs=q,hardware=h,joint=f,ideal_joint=ideal_f,sample_ids=m.iloc[c].sample_id.to_numpy(dtype=str))
        seen[c]+=1
        for name,head,features in [('linear16',bundle['linear_head'],q),('ideal256_matched',bundle['ideal_head'],ideal_f),('hardware256',bundle['hardware_head'],f)]:
            ss=head.predict(features);pred=ss.argmax(axis=1);pooled[name][c]=pred
            for pos,ix in enumerate(c):
                predictions.append(dict(fold=fold,model=name,sample_id=m.iloc[ix].sample_id,user_id=m.iloc[ix].user_id,true_label=CLASSES[y[ix]],predicted_label=CLASSES[pred[pos]],**{f'score_{cl}':float(ss[pos,j]) for j,cl in enumerate(CLASSES)}))
    assert (seen==1).all()
    pd.DataFrame(predictions).to_csv(OUT/'outer_predictions.csv',index=False)
    rows=[];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(name,pred) in zip(axes,pooled.items()):
        result=dict(**scores(y,pred),per_class=classification_report(y,pred,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0),confusion_counts=confusion_matrix(y,pred,labels=range(4)).tolist(),confusion_normalized=confusion_matrix(y,pred,labels=range(4),normalize='true').tolist())
        save(OUT/f'{name}_metrics.json',result);rows.append(dict(model=name,**scores(y,pred)))
        cm=np.array(result['confusion_counts']);ax.imshow(cm,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,title=name,xlabel='Predicted',ylabel='True')
    fig.tight_layout();fig.savefig(OUT/'confusion_counts.png',dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
    save(OUT/'diagnostics.json',diagnostics)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    save(OUT/'verification.json',dict(sources_unchanged=True,test_evaluated=False,all_outer_samples_predicted_once=True,elapsed_seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,torch=torch.__version__))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
