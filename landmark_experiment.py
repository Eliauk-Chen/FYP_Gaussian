"""Development-only 2D hand-landmark classification; preserve all previous studies."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler, KernelCenterer
from threadpoolctl import threadpool_limits
from multiwidth_experiment import save, read, score, metrics, phi, cm_plot, snapshot

CLASSES=['fist','like','peace','ok']
ALPHAS=[.001,.01,.1,1.,10.,100.]
FACTORS=[.25,.5,1.,2.,4.]
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def features_and_audit(out):
    m=pd.read_csv('artifacts/hagrid_followup_v1/fresh_manifest.csv')
    # Only train/val annotation files are opened; old test samples are not evaluated.
    dev=m[m.split.ne('test')].copy(); features=[]; accepted=[]; audit=[]; examples={}
    for (split,label),rows in dev.groupby(['split','class'],sort=False):
        annotations=read(Path('data/official_hagrid_audit/annotations')/split/(label+'.json'))
        for original_index,row in rows.iterrows():
            ann=annotations[row.image_id];idx=int(row.hand_index)
            assert ann['user_id']==row.user_id and ann['labels'][idx]==label
            assert np.allclose(ann['bboxes'][idx],json.loads(row.bbox_norm_xywh))
            record=dict(sample_id=row.sample_id,original_index=int(original_index),split=split,label=label,accepted=False,reason='')
            hands=ann.get('hand_landmarks')
            if not isinstance(hands,list) or len(hands)!=len(ann['bboxes']):
                record['reason']='missing_or_unaligned_hand_arrays';audit.append(record);continue
            try: points=np.asarray(hands[idx],dtype=float)
            except (ValueError,TypeError): points=np.empty((0,))
            if points.shape!=(21,2) or not np.isfinite(points).all():
                record['reason']='not_21_finite_2D_landmarks';audit.append(record);continue
            x,y,w,h=ann['bboxes'][idx]
            inside=((points[:,0]>=x-.1*w)&(points[:,0]<=x+1.1*w)&(points[:,1]>=y-.1*h)&(points[:,1]<=y+1.1*h)).mean()
            record['fraction_in_expanded_target_box']=float(inside)
            if inside<.5:
                record['reason']='less_than_half_points_inside_expanded_target_box';audit.append(record);continue
            with Image.open(row.path) as im: width,height=im.size
            pixels=points*np.array([width,height])
            offsets=pixels-pixels[0]
            scale=float(np.sqrt(np.mean(np.sum(offsets[1:]**2,axis=1))))
            record['scale_pixels']=scale
            if scale<=1e-6:
                record['reason']='degenerate_scale';audit.append(record);continue
            feature=(offsets/scale).ravel()
            record.update(accepted=True,reason='accepted');audit.append(record)
            features.append(feature);item=row.to_dict();item['original_index']=int(original_index);accepted.append(item)
            if len(examples.get(label,[]))<4:examples.setdefault(label,[]).append((row,pixels))
    manifest=pd.DataFrame(accepted).sort_values('original_index')
    indexed={r['original_index']:f for r,f in zip(accepted,features)}
    x=np.array([indexed[i] for i in manifest.original_index],dtype=np.float64)
    assert manifest.user_id.is_unique and np.isfinite(x).all()
    manifest.to_csv(out/'development_manifest.csv',index=False)
    pd.DataFrame(audit).sort_values('original_index').to_csv(out/'landmark_audit.csv',index=False)
    np.savez_compressed(out/'landmark_features.npz',features=x,sample_ids=manifest.sample_id.to_numpy(),labels=manifest['class'].to_numpy(),user_ids=manifest.user_id.to_numpy())
    counts=pd.DataFrame(audit).groupby(['split','label']).accepted.agg(['size','sum']).rename(columns={'size':'available','sum':'accepted'})
    counts['excluded']=counts.available-counts.accepted;counts['retention']=counts.accepted/counts.available;counts.to_csv(out/'retention.csv')
    # Deterministic first four accepted examples per class, no prediction selection.
    grid=Image.new('RGB',(4*320,4*300),'white')
    for ci,label in enumerate(CLASSES):
        for j,(row,pixels) in enumerate(examples.get(label,[])):
            im=Image.open(row.path).convert('RGB');draw=ImageDraw.Draw(im)
            for a,b in EDGES:draw.line([tuple(pixels[a]),tuple(pixels[b])],fill='lime',width=2)
            for k,(px,py) in enumerate(pixels):draw.ellipse((px-2,py-2,px+2,py+2),fill='red')
            bx,by,bw,bh=json.loads(row.bbox_norm_xywh);ww,hh=im.size
            box=(max(0,int((bx-.3*bw)*ww)),max(0,int((by-.3*bh)*hh)),min(ww,int((bx+1.3*bw)*ww)),min(hh,int((by+1.3*bh)*hh)))
            crop=ImageOps.contain(im.crop(box),(310,265));grid.paste(crop,(j*320,ci*300))
            ImageDraw.Draw(grid).text((j*320,ci*300+268),f'{label} {row.split}\n{row.image_id[:25]}',fill='black')
    grid.save(out/'landmark_preview.jpg')
    save(out/'data_audit.json',dict(source='official HaGRIDv2 automatic hand_landmarks',coordinates='21 x 2 normalized full-image xy; no z provided',
        available_development=800,accepted=len(manifest),excluded=800-len(manifest),users=int(manifest.user_id.nunique()),
        automatic_labels_not_manual_ground_truth=True,test_evaluated=False,normalization='Pixel x/y (aspect corrected), subtract wrist index 0, divide by RMS distance of other 20 points to wrist. No rotation/reflection canonicalization; no PCA.',
        feature_dimensions=42,quality_rule='21x2 finite points; aligned bbox/label/landmark array; >=50% points in target bbox expanded 10%; nondegenerate scale.',
        limitations=['Automatic landmarks may be inaccurate even after structural audit.','Missing landmarks excluded equally for all classifiers; evaluate retention and conditional classification separately.','Previous development data reused: exploratory, not new confirmation.']))
    return manifest,x


def fit_compare(xtr,xval,ytr,yval,xhold,yhold,rows,path):
    path.mkdir()
    scaler=StandardScaler().fit(xtr);tr=scaler.transform(xtr);va=scaler.transform(xval);ho=scaler.transform(xhold)
    km=KMeans(n_clusters=32,random_state=42,n_init=10).fit(tr)
    distances=np.linalg.norm(tr[:,None]-km.cluster_centers_[None],axis=2)
    ref=max(float(np.median(distances[distances>0])),1e-8);pool=np.array(FACTORS)*ref
    joblib.dump(dict(scaler=scaler,kmeans=km,sigma_ref=ref),path/'shared.joblib')
    records=[];best={};models={}
    def consider(name,model,xx,alpha,sigma=None,extra=None):
        scores=model.predict(xx)
        if name=='K_full_gaussian':pred=np.array(CLASSES)[np.argmax(scores+extra['target_mean'],axis=1)]
        else:pred=scores
        f=score(yval,pred);r=dict(model=name,alpha=alpha,sigma=None if sigma is None else float(sigma),val_macro_f1=f)
        records.append(r)
        if name not in best or f>best[name]['val_macro_f1']:
            best[name]=r;models[name]=(model,extra)
    for alpha in ALPHAS:
        model=RidgeClassifier(alpha=alpha).fit(tr,ytr);consider('A_linear',model,va,alpha)
    targets=np.column_stack([ytr==c for c in CLASSES]).astype(float);mean=targets.mean(axis=0)
    for sigma in pool:
        rtr=phi(tr,km.cluster_centers_,[sigma]*32);rva=phi(va,km.cluster_centers_,[sigma]*32)
        gamma=1/(2*sigma**2);ktr=rbf_kernel(tr,tr,gamma=gamma);center=KernelCenterer().fit(ktr)
        ktr=center.transform(ktr);kva=center.transform(rbf_kernel(va,tr,gamma=gamma))
        for alpha in ALPHAS:
            model=RidgeClassifier(alpha=alpha).fit(rtr,ytr);consider('B_rbf32',model,rva,alpha,sigma)
            model=KernelRidge(alpha=alpha,kernel='precomputed').fit(ktr,targets-mean)
            consider('K_full_gaussian',model,kva,alpha,sigma,dict(centerer=center,train_features=tr,target_mean=mean,gamma=float(gamma)))
    save(path/'validation_search.json',records);save(path/'locked_configs.json',dict(configs=best,locked_utc=datetime.now(timezone.utc).isoformat()))
    result=[]
    for name,cfg in best.items():
        model,extra=models[name];joblib.dump(dict(model=model,extra=extra,config=cfg),path/(name+'.joblib'))
        if name=='A_linear':pred=model.predict(ho)
        elif name=='B_rbf32':pred=model.predict(phi(ho,km.cluster_centers_,[cfg['sigma']]*32))
        else:
            kh=extra['centerer'].transform(rbf_kernel(ho,tr,gamma=extra['gamma']))
            pred=np.array(CLASSES)[(model.predict(kh)+extra['target_mean']).argmax(axis=1)]
        met=metrics(yhold,pred);save(path/(name+'_metrics.json'),met);cm_plot(met,name,path)
        frame=rows[['sample_id','user_id','class']].copy();frame['predicted_label']=pred;frame.to_csv(path/(name+'_outer_predictions.csv'),index=False)
        result.append(dict(model=name,inner_val_macro_f1=cfg['val_macro_f1'],**{k:met[k] for k in ['accuracy','macro_f1','balanced_accuracy']},alpha=cfg['alpha'],sigma=cfg['sigma'],training_n=len(ytr),outer_n=len(yhold)))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_landmarks_v1'));args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    sources=[Path('artifacts/hagrid_small'),Path('artifacts/hagrid_multiwidth_v1'),Path('artifacts/hagrid_followup_v1')]
    hashes={str(p):snapshot(p) for p in sources};save(out/'previous_hashes.json',hashes)
    save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),classes=CLASSES,alphas=ALPHAS,factors=FACTORS,
       models=['A_linear','B_rbf32','K_full_gaussian'],feature='42D normalized 2D landmark coordinates; no PCA',
       quality_filter='Finite 21x2; array alignment; >=50% points in target bbox with 10% context; RMS >1e-6 pixels',
       split='Reuse prior 3 nested development folds, intersect with valid landmarks; no test evaluation',
       kernel='Centered Gaussian Gram matrix + centered one-hot targets + KernelRidge; add training target mean back for class scores.',
       kernel_budget='Full kernel uses every training sample as a reference; not a 32-node, matched-cost comparison.',
       model_selection='Inner-validation Macro-F1 only; ties retain first candidate; all three models evaluated on outer fold.'))
    m,x=features_and_audit(out);lookup={int(i):j for j,i in enumerate(m.original_index)};results=[]
    with threadpool_limits(limits=1):
        for fold in range(3):
            original=read(Path('artifacts/hagrid_followup_v1')/f'cv_fold_{fold}'/'split_indices.json')
            indices={k:[lookup[i] for i in v if i in lookup] for k,v in original.items()}
            a,b,c=(np.array(indices[k]) for k in ['train','val','outer_holdout'])
            for ii in [a,b,c]:assert set(m.iloc[ii]['class'])==set(CLASSES)
            assert not set(m.iloc[a].user_id)&set(m.iloc[b].user_id)
            assert not set(m.iloc[np.r_[a,b]].user_id)&set(m.iloc[c].user_id)
            assert m.iloc[np.r_[a,b,c]].split.ne('test').all()
            path=out/f'fold_{fold}'
            r=fit_compare(x[a],x[b],m.iloc[a]['class'].to_numpy(),m.iloc[b]['class'].to_numpy(),x[c],m.iloc[c]['class'].to_numpy(),m.iloc[c],path)
            save(path/'split_indices.json',indices)
            for v in r:v['fold']=fold
            results.extend(r);print('COMPLETED FOLD',fold,flush=True)
    table=pd.DataFrame(results);table.to_csv(out/'outer_results.csv',index=False)
    summary=table.groupby('model').agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    for name in summary.model:
        frames=[pd.read_csv(out/f'fold_{i}'/(name+'_outer_predictions.csv')) for i in range(3)]
        oof=pd.concat(frames,ignore_index=True);assert oof.sample_id.is_unique and len(oof)==len(m)
        oof.to_csv(out/(name+'_oof_predictions.csv'),index=False);save(out/(name+'_pooled_oof_metrics.json'),metrics(oof['class'],oof.predicted_label))
    save(out/'verification.json',dict(previous_files_unchanged=hashes=={str(p):snapshot(p) for p in sources},test_evaluated=False,development_n=len(m),feature_dimensions=42,folds=3))
    assert read(out/'verification.json')['previous_files_unchanged']
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'versions.json',{p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','pandas','joblib','Pillow']})
    print(summary.to_string(index=False));print('AUDIT',read(out/'data_audit.json'))


if __name__=='__main__':main()
