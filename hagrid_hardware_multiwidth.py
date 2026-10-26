"""Node-wise target widths on the fixed HaGRID empirical-response pipeline."""
import argparse, copy, hashlib, json, shutil, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from hagrid_hardware_clean import CLASSES, ALPHAS, save, scores, FullJointGaussianTransformer
from hagrid_hardware_width_scan import BASE, CACHE, POOL, response
from landmark_failure_multiwidth import variants, KINDS
import matplotlib.pyplot as plt

SINGLE=Path('artifacts/hagrid_hardware_width_scan_v1')

def select_columns(cache, indices):
    return cache[indices,:,np.arange(256)].T.copy()

def macro_by_level(y, pred):
    f=[]
    for c in range(4):
        tp=((pred==c)&(y[None,:]==c)).sum(axis=1)
        den=(pred==c).sum(axis=1)+(y==c).sum()
        f.append(np.divide(2*tp,den,out=np.zeros(len(pred)),where=den!=0))
    return np.mean(f,axis=0)

def transform_from_bundle(bundle):
    # Only transform() is used: scalar joint_sigma is irrelevant once weights exist.
    # Keep vector semantics explicit in the surrounding bundle, not a scalar-width checkpoint.
    state=copy.deepcopy(bundle['shared_transform_state'])
    state['calibration_weights']=bundle['calibration_weights']
    return FullJointGaussianTransformer.from_state_dict(state)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_hardware_multiwidth_v1'))
    ap.add_argument('--seconds-per-search',type=float,default=90)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    files=[p for root in [BASE,SINGLE] for p in root.rglob('*') if p.is_file()]
    files+=list(CACHE.glob('corruptions_seed_*.npz'))+[Path('hagrid_hardware_width_scan.py'),Path('landmark_failure_multiwidth.py'),Path('artifacts/hagrid_landmarks_v1/landmark_features.npz')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};save(out/'source_hashes.json',hashes)
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'protocol.json',dict(width_pool=POOL,alphas=ALPHAS,nodes=256,output_features=256,
        fixed='Input preprocessing, empirical mean-response bank, splits, 256 centers, calibration alpha and clipping bounds from clean v2.',
        fit='Clean inner training only. Precompute calibration columns for each target sigma; combine exactly one sigma per node across all 16 input dimensions.',
        validation='Separate configuration per corruption kind; seeds300+fold; equal severity mean Macro-F1 including clean, original conditions.',
        directed=dict(initialization='Previously selected shared width',node_order_seed=42,max_sweeps=2,seconds_per_search=args.seconds_per_search,acceptance='Strictly higher validation Macro-F1 only; equal scores retain current configuration'),
        random=dict(seeds=list(range(10)),distribution='Each of 256 width indices independently uniform on the same pool; all ten reported, no winner selected'),
        baseline='Reuse saved shared-width head and transform; re-evaluate validation only to verify parity.',
        evaluation='Lock all 9 cases before outer cache evaluation; original corruption seeds0..9; no official test. Reused development users, exploratory.',
        budget='Directed uses more validation feedback than 10 random configurations; not budget matched. No matched-budget random search in this round.',
        scope='Software calibration target widths, not physical channel widths. No width concatenation, no extra device noise, no Linear, no augmentation.'))
    m=pd.read_csv(BASE/'manifest.csv');m.to_csv(out/'manifest.csv',index=False)
    x=np.load('artifacts/hagrid_landmarks_v1/landmark_features.npz',allow_pickle=True)['features']
    y=np.array([CLASSES.index(c) for c in m['class']]);assert m.split.ne('test').all()
    audit=[];locked=[];allmodels=[];seen=np.zeros(len(x),int)
    for fold in range(3):
        b=joblib.load(BASE/f'fold_{fold}/model_bundle.joblib');prior=joblib.load(SINGLE/f'fold_{fold}/models.joblib')
        ids=json.loads((BASE/f'fold_{fold}/split_indices.json').read_text());a,v,c=[np.array(ids[k]) for k in ['train','val','outer_holdout']]
        us=[set(m.iloc[ix].user_id) for ix in [a,v,c]];assert not(us[0]&us[1] or us[0]&us[2] or us[1]&us[2]);seen[c]+=1
        qa,ha=response(b,x[a]);original=FullJointGaussianTransformer.from_state_dict(b['transform_state'])
        transforms=[]
        for sigma in POOL:
            tr=FullJointGaussianTransformer(16,16,256,sigma,original.calibration_alpha,original.factor_lower,original.factor_upper,original.epsilon,42,300).fit(ha,qa,preset_centers=original.centers_)
            np.testing.assert_array_equal(tr.centers_,original.centers_)
            np.testing.assert_allclose(tr.channel_mean_,original.channel_mean_);np.testing.assert_allclose(tr.channel_std_,original.channel_std_)
            transforms.append(tr)
        traincache=np.array([tr.transform(ha).astype(float) for tr in transforms])
        foldcases={}
        for kind,levels in KINDS.items():
            folder=out/f'fold_{fold}'/kind;folder.mkdir(parents=True)
            validation=variants(x[v],kind,300+fold,x[a].mean(axis=0));_,hv=response(b,validation.reshape(-1,42))
            valcache=np.array([tr.transform(hv).astype(float) for tr in transforms])
            old=prior['models'][kind]['width_tuned'];index=POOL.index(old['config']['sigma']);initial=np.full(256,index,int)
            oldtr=FullJointGaussianTransformer.from_state_dict(old['transform_state'])
            np.testing.assert_allclose(traincache[index],oldtr.transform(ha),rtol=0,atol=0)
            trials=[];fit_count=0
            def evaluate(indices,label):
                nonlocal fit_count
                t=time.monotonic();ft=select_columns(traincache,indices);fv=select_columns(valcache,indices);best=None;grid=[]
                for alpha in ALPHAS:
                    head=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(ft,np.eye(4)[y[a]]);fit_count+=1
                    pred=head.predict(fv).argmax(axis=1).reshape(len(levels),len(v));perlevel=macro_by_level(y[v],pred);obj=float(perlevel.mean())
                    grid.append(dict(alpha=alpha,objective=obj,per_level_f1=perlevel.tolist()))
                    if best is None or obj>best['objective']:best=dict(objective=obj,alpha=alpha,head=head,indices=indices.copy(),per_level_f1=perlevel.tolist())
                rec=dict(id=len(trials),label=label,indices=indices.tolist(),alpha_grid=grid,best_alpha=best['alpha'],objective=best['objective'],seconds=time.monotonic()-t)
                trials.append(rec);best['trial_id']=rec['id'];return best
            baseline=evaluate(initial,'shared_replay')
            assert abs(baseline['objective']-old['config']['val_objective'])<1e-12
            np.testing.assert_allclose(baseline['head'].coef_,old['head'].coef_)
            randoms=[evaluate(np.random.default_rng(seed).integers(0,len(POOL),256),f'random_{seed}') for seed in range(10)]
            current=baseline;accepted=[];orders=[];search_start=time.monotonic();search_fits=fit_count;search_trials=len(trials);timeout=False
            rng=np.random.default_rng(42)
            for sweep in range(2):
                changed=False;order=rng.permutation(256);orders.append(order.tolist())
                for node in order:
                    winner=current
                    for k in range(len(POOL)):
                        if k==current['indices'][node]:continue
                        if time.monotonic()-search_start>=args.seconds_per_search:timeout=True;break
                        indices=current['indices'].copy();indices[node]=k
                        result=evaluate(indices,f'sweep{sweep}_node{node}_width{k}')
                        if result['objective']>winner['objective']:winner=result
                    if winner['objective']>current['objective']:
                        accepted.append(dict(sweep=sweep,node=int(node),old_sigma=POOL[current['indices'][node]],new_sigma=POOL[winner['indices'][node]],before=current['objective'],after=winner['objective'],trial_id=winner['trial_id']))
                        current=winner;changed=True
                    if timeout:break
                if timeout or not changed:break
            configs={'single':baseline,'directed':current,**{f'random_{i}':r for i,r in enumerate(randoms)}}
            saved={}
            for name,r in configs.items():
                weights=np.empty_like(original.calibration_weights_)
                for j,k in enumerate(r['indices']):weights[:,:,j]=transforms[k].calibration_weights_[:,:,j]
                bundle=dict(type='per_node_target_width_hardware_joint_rbf',shared_transform_state=original.state_dict(),calibration_weights=weights,
                            widths=np.array(POOL)[r['indices']],head=r['head'],classes=CLASSES,config={k:v for k,v in r.items() if k not in ['head','indices']})
                tr=transform_from_bundle(bundle)
                np.testing.assert_allclose(tr.transform(ha),select_columns(traincache,r['indices']),atol=0,rtol=0)
                np.testing.assert_allclose(tr.transform(hv),select_columns(valcache,r['indices']),atol=0,rtol=0)
                saved[name]=bundle
            joblib.dump(saved,folder/'models.joblib');save(folder/'trials.json',trials);save(folder/'accepted_moves.json',accepted)
            search=dict(fold=fold,kind=kind,single_validation=baseline['objective'],directed_validation=current['objective'],random_validation=[r['objective'] for r in randoms],
                        random_mean=float(np.mean([r['objective'] for r in randoms])),random_sd=float(np.std([r['objective'] for r in randoms],ddof=1)),
                        search_seconds=time.monotonic()-search_start,search_configurations=len(trials)-search_trials,search_ridge_fits=fit_count-search_fits,
                        total_ridge_fits=fit_count,timeout=timeout,accepted_moves=len(accepted),node_orders=orders)
            save(folder/'search_summary.json',search);locked.append(search);foldcases[kind]=saved
            for name,r in configs.items():audit.append(dict(fold=fold,kind=kind,model=name,validation_objective=r['objective'],alpha=r['alpha']))
            save(out/'validation_progress.json',locked)
            print('LOCKED',fold,kind,'single',round(baseline['objective'],5),'directed',round(current['objective'],5),'configs',len(trials)-search_trials,'timeout',timeout,flush=True)
        save(out/f'fold_{fold}/split_indices.json',ids);joblib.dump(b,out/f'fold_{fold}/shared.joblib');allmodels.append((b,c,foldcases))
    assert (seen==1).all();save(out/'locked_configs.json',locked);pd.DataFrame(audit).to_csv(out/'validation_results.csv',index=False)
    names=['single','directed']+[f'random_{i}' for i in range(10)];rows=[];classes=[];matrices=[]
    for noise_seed in range(10):
        with np.load(CACHE/f'corruptions_seed_{noise_seed}.npz') as z:
            assert np.array_equal(z['sample_ids'],m.sample_id)
            keys=z['keys'].copy();predictions=np.empty((len(keys),len(names),len(x)),dtype=np.int8);all_scores=np.empty((len(keys),len(names),len(x),4),dtype=np.float64)
            for ki,(key,xx) in enumerate(zip(keys,z['features'])):
                kind,level=key.split('_');level=float(level)
                for b,idx,cases in allmodels:
                    _,h=response(b,xx[idx])
                    for ni,name in enumerate(names):
                        model=cases[kind][name];tr=transform_from_bundle(model);ft=tr.transform(h).astype(float);assert np.isfinite(ft).all()
                        ss=model['head'].predict(ft);all_scores[ki,ni,idx]=ss;predictions[ki,ni,idx]=ss.argmax(axis=1)
                for ni,name in enumerate(names):
                    yp=predictions[ki,ni];tag=dict(noise_seed=noise_seed,kind=kind,level=level,model=name)
                    rows.append(dict(**tag,**scores(y,yp)))
                    rep=classification_report(y,yp,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0)
                    for cl in CLASSES:classes.append(dict(**tag,label=cl,**rep[cl]))
                    matrices.append(dict(**tag,counts=confusion_matrix(y,yp,labels=range(4)).tolist(),normalized_true=confusion_matrix(y,yp,labels=range(4),normalize='true').tolist()))
        np.savez_compressed(out/f'predictions_noise_seed_{noise_seed}.npz',condition_keys=keys,models=np.array(names),sample_ids=m.sample_id.to_numpy(dtype=str),user_ids=m.user_id.to_numpy(dtype=str),classes=np.array(CLASSES),true_labels=y,predicted_labels=predictions,class_scores=all_scores)
        print('EVALUATED NOISE SEED',noise_seed,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics_all_seeds.csv',index=False);pd.DataFrame(classes).to_csv(out/'per_class.csv',index=False);save(out/'confusions.json',matrices)
    prev=pd.read_csv(SINGLE/'seed_metrics.csv');prev=prev[prev.model=='width_tuned']
    joined=df[df.model=='single'].merge(prev,on=['kind','level'],suffixes=('_new','_old'));joined=joined[joined.noise_seed==joined.seed]
    for metric in ['accuracy','macro_f1','balanced_accuracy']:np.testing.assert_allclose(joined[metric+'_new'],joined[metric+'_old'],atol=1e-14,rtol=0)
    summary=df.groupby(['kind','level','model']).agg(accuracy_mean=('accuracy','mean'),macro_f1_mean=('macro_f1','mean'),noise_seed_sd=('macro_f1','std'),balanced_accuracy_mean=('balanced_accuracy','mean')).reset_index();summary.to_csv(out/'summary_per_model.csv',index=False)
    random=summary[summary.model.str.startswith('random_')].groupby(['kind','level']).agg(accuracy_mean=('accuracy_mean','mean'),macro_f1_mean=('macro_f1_mean','mean'),width_seed_sd=('macro_f1_mean','std'),balanced_accuracy_mean=('balanced_accuracy_mean','mean')).reset_index();random['model']='random_mean';random.to_csv(out/'random_summary.csv',index=False)
    comparison=pd.concat([summary[summary.model.isin(['single','directed'])],random],ignore_index=True);comparison.to_csv(out/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,kind in zip(axes,KINDS):
        for name,g in comparison[comparison.kind==kind].groupby('model'):ax.plot(g.level,g.macro_f1_mean,'o-',label=name)
        ax.set(title=kind,xlabel='Noise SD / RMS' if kind=='jitter' else 'Affected points /20',ylabel='Macro-F1',ylim=(0,1.03));ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for fold in range(3):
        for j,kind in enumerate(KINDS):
            moves=json.loads((out/f'fold_{fold}/{kind}/accepted_moves.json').read_text());info=next(r for r in locked if r['fold']==fold and r['kind']==kind)
            ax=axes[fold,j];ax.step([0]+[r['trial_id'] for r in moves],[info['single_validation']]+[r['after'] for r in moves],where='post');ax.set(title=f'fold {fold} / {kind}',xlabel='Candidate trial ID',ylabel='Validation Macro-F1');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'directed_search.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for fold,(_,_,cases) in enumerate(allmodels):
        for j,kind in enumerate(KINDS):
            ax=axes[fold,j];w=cases[kind]['directed']['widths'];ax.bar(range(len(POOL)),[(w==v).sum() for v in POOL]);ax.set(xticks=range(len(POOL)),xticklabels=POOL,title=f'fold {fold} / {kind}',ylabel='Nodes',xlabel='Target sigma')
    fig.tight_layout();fig.savefig(out/'width_distribution.png',dpi=160);plt.close(fig)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out/'verification.json',dict(sources_unchanged=True,single_baseline_metrics_match=True,column_composition_exact=True,user_disjoint=True,official_test_used=False,output_dimensions=256,elapsed_seconds=time.monotonic()-start))
    print(comparison[comparison.level.isin([0,.2,.7,14])].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
