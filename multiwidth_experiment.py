"""Frozen-center multiwidth RBF study. Never writes to the baseline directory.

Stages run in order: audit -> validation-only selection -> configuration lock
-> saved-baseline diagnostics -> one evaluation of all prescribed new models.
"""
import argparse
import hashlib
import json
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
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

CLASSES = ['fist', 'like', 'peace', 'ok']
METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy']


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(base):
    return {str(p.relative_to(base)): sha(p) for p in sorted(base.rglob('*')) if p.is_file()}


def score(y, pred):
    return float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0))


def phi(z, centers, widths):
    # Preserve original float32 response arithmetic, one column per center.
    d2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    x = np.column_stack([np.exp(-d2[:, j] / (2 * float(w) ** 2)) for j, w in enumerate(widths)])
    assert np.isfinite(x).all() and not np.any(np.all(x == 0, axis=1))
    return x


def audit(base, out):
    manifest = pd.read_csv(base / 'manifest.csv', keep_default_na=False)
    cache = np.load(base / 'features_resnet18.npz')
    # Do not access cache['labels'] here: use only training/validation targets.
    assert np.array_equal(cache['sample_ids'], manifest.sample_id)
    assert np.array_equal(cache['splits'], manifest.split)
    assert manifest.sample_id.is_unique
    assert manifest.groupby('image_id').split.nunique().max() == 1
    features = cache['features']
    assert np.isfinite(features).all()
    prep = joblib.load(base / 'preprocessing.joblib')
    old = joblib.load(base / 'rbf_ridge.joblib')
    search = read(base / 'validation_search.json')
    train = manifest.split.eq('train').to_numpy()
    val = manifest.split.eq('val').to_numpy()
    sc, pca, km = prep['scaler'], prep['pca'], old['kmeans']
    assert sc.n_samples_seen_ == train.sum() == pca.n_samples_
    assert np.allclose(sc.mean_, features[train].mean(axis=0, dtype=np.float64))
    assert np.allclose(sc.var_, features[train].var(axis=0, dtype=np.float64))
    scaled = sc.transform(features[train])
    assert np.allclose(pca.mean_, scaled.mean(axis=0), atol=1e-6)
    ztr, zval = pca.transform(scaled), pca.transform(sc.transform(features[val]))
    assert len(km.labels_) == train.sum()
    center_error = max(float(np.max(np.abs(ztr[km.labels_ == j].mean(axis=0) - c))) for j, c in enumerate(km.cluster_centers_))
    assert center_error < 1e-3
    ref = max(float(np.median(np.linalg.norm(ztr[:, None] - km.cluster_centers_[None], axis=2)[np.linalg.norm(ztr[:, None] - km.cluster_centers_[None], axis=2) > 0])), 1e-8)
    assert np.isclose(ref, old['sigma_ref'])
    # Verify publisher source_split and original annotations against every manifest row.
    source_checks = 0
    for path, rows in manifest.groupby('source_parquet'):
        table = pq.read_table(path, columns=['source_split', 'image_id', 'hagridv2_original']).to_pylist()
        for _, row in rows.iterrows():
            src = table[int(row.row_index)]
            assert {'valid': 'val'}.get(src['source_split'], src['source_split']) == row.split
            assert str(src['image_id']) == str(row.image_id)
            ann = json.loads(src['hagridv2_original'])
            assert np.allclose(ann['bboxes'][int(row.hand_index)], json.loads(row.bbox_norm_xywh))
            source_checks += 1
    ytr = manifest.loc[train, 'class'].to_numpy()
    yval = manifest.loc[val, 'class'].to_numpy()
    linear = joblib.load(base / 'linear_ridge.joblib')
    assert np.isclose(score(yval, linear.predict(zval)), search['best_linear']['val_macro_f1'])
    baseline_phi = phi(ztr, km.cluster_centers_, [old['sigma']] * km.n_clusters)
    assert np.isclose(score(yval, old['ridge'].predict(phi(zval, km.cluster_centers_, [old['sigma']] * km.n_clusters))), search['best_rbf']['val_macro_f1'])
    counts = manifest.groupby(['split', 'class']).size().unstack().reindex(columns=CLASSES)
    counts.to_csv(out / 'sample_counts.csv')
    encoding = read(base / 'feature_extraction_config.json')
    report = dict(data=read(base / 'data_audit.json'), counts=counts.to_dict(orient='index'),
                  users={s: {'known_user_count': int(manifest.loc[manifest.split.eq(s) & manifest.user_id.ne(''), 'user_id'].nunique()), 'actual_user_count': None} for s in ['train', 'val', 'test']},
                  user_intersections=None, cross_user_verified=False,
                  score_set='test', selection_set='val', same_samples_and_features=True,
                  train_only_scaler_pca_kmeans=True, center_mean_max_error=center_error,
                  encoder=encoding, pca_dimensions=int(pca.n_components_), pca_whiten=pca.whiten,
                  kmeans_parameters=km.get_params(), sigma_ref=ref, best_linear=search['best_linear'], best_single=search['best_rbf'],
                  source_rows_verified=source_checks, missing_user_ids=int(manifest.user_id.eq('').sum()),
                  issues=['user_id omitted by publisher; actual user counts and overlap are unknown.',
                          'Original script inferred splits from filenames; this audit additionally verified source_split for every selected row.',
                          'Subset is MediaPipe quality-filtered, not a random sample of full HaGRIDv2.',
                          'Original encoder transform uses 256 resize and 224 center crop; finger retention was not quantitatively verified before original test. Frozen features retained as requested.'],
                  fixes='Added source_split verification and explicit checks; no baseline model, feature, split or score altered.')
    save(out / 'baseline_audit.json', report)
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    save(out / 'train_rbf_responses.json', dict(quantiles=dict(zip(map(str, quantiles), map(float, np.quantile(baseline_phi, quantiles)))),
         node_std=baseline_phi.std(axis=0).tolist(), near_zero_threshold=1e-6,
         near_one_threshold=1-1e-6, fraction_near_zero=float((baseline_phi < 1e-6).mean()),
         fraction_near_one=float((baseline_phi > 1-1e-6).mean()), finite=True, all_zero_rows=0))
    pd.DataFrame({'node': np.arange(km.n_clusters), 'std': baseline_phi.std(axis=0),
                  'mean': baseline_phi.mean(axis=0), 'min': baseline_phi.min(axis=0), 'max': baseline_phi.max(axis=0)}).to_csv(out / 'node_response_stats.csv', index=False)
    for file in ['preprocessing.joblib', 'rbf_ridge.joblib', 'linear_ridge.joblib', 'validation_search.json', 'feature_extraction_config.json']:
        shutil.copy2(base / file, out / ('baseline_' + file))
    return manifest, features, ztr, zval, ytr, yval, old, search, prep


class Search:
    def __init__(self, ztr, zval, ytr, yval, centers, pool, alphas, out):
        self.pool, self.alphas, self.ytr, self.yval = pool, alphas, ytr, yval
        self.m, self.out, self.records, self.fits = len(centers), out, [], 0
        self.tr = np.stack([phi(ztr, centers, [w]*self.m) for w in pool], axis=2)
        self.va = np.stack([phi(zval, centers, [w]*self.m) for w in pool], axis=2)

    def evaluate(self, indices, group, deadline=None):
        if deadline and time.perf_counter() >= deadline: return None
        start = time.perf_counter()
        xtr, xva = self.tr[:, np.arange(self.m), indices], self.va[:, np.arange(self.m), indices]
        best, results = None, []
        for alpha in self.alphas:
            if deadline and time.perf_counter() >= deadline:
                self.records.append(dict(group=group, complete=False, width_indices=list(map(int, indices)), alpha_results=results, elapsed_seconds=time.perf_counter()-start))
                return None
            model = RidgeClassifier(alpha=alpha).fit(xtr, self.ytr)
            self.fits += 1
            f = score(self.yval, model.predict(xva))
            results.append({'alpha': alpha, 'val_macro_f1': f})
            if best is None or f > best['val_macro_f1']:
                best = dict(alpha=alpha, val_macro_f1=f)
        best.update(group=group, complete=True, width_indices=list(map(int, indices)),
                    widths=[float(self.pool[i]) for i in indices], alpha_results=results,
                    elapsed_seconds=time.perf_counter()-start, evaluation_id=len(self.records))
        self.records.append(best)
        return best

    def fit_final(self, config):
        x = self.tr[:, np.arange(self.m), config['width_indices']]
        return RidgeClassifier(alpha=config['alpha']).fit(x, self.ytr)


def select(out, ztr, zval, ytr, yval, old, original):
    pool = sorted({float(r['sigma']) for r in original['rbf']})
    alphas = sorted({float(r['alpha']) for r in original['linear']})
    assert min(pool) > 0 and np.isfinite(pool).all()
    protocol = dict(pool=pool, alphas=alphas, nodes=len(old['kmeans'].cluster_centers_),
                    coordinate_seed=42, max_rounds=2, coordinate_budget_seconds=900,
                    random_seeds=list(range(10)), matched_random_seed=10042,
                    match_rule='Match full coordinate configuration evaluations INCLUDING initialization; six Ridge fits per configuration.',
                    coordinate_rule='Fixed shuffled node order reused each round; examine four alternatives to node value at visit start; accept best strict improvement; ties keep incumbent.',
                    test_models=['A_linear_reuse', 'B_single_reuse', 'C_directed'] + [f'D_random_{i}' for i in range(10)] + ['E_budget_matched'],
                    evaluation_status='Exploratory reuse of same held-out test; not independent confirmation.')
    save(out / 'protocol.json', protocol)
    engine = Search(ztr, zval, ytr, yval, old['kmeans'].cluster_centers_, pool, alphas, out)
    random_configs = []
    for seed in range(10):
        indices = np.random.default_rng(seed).integers(0, len(pool), size=engine.m)
        c = engine.evaluate(indices, f'D_random_{seed}'); c['seed'] = seed
        random_configs.append(c)
    start, fits0, records0 = time.perf_counter(), engine.fits, len(engine.records)
    deadline = start + 900
    current = np.full(engine.m, pool.index(old['sigma']), dtype=int)
    best = engine.evaluate(current, 'C_initial', deadline)
    assert np.isclose(best['val_macro_f1'], original['best_rbf']['val_macro_f1'])
    order = np.random.default_rng(42).permutation(engine.m).tolist()
    accepted, rounds, timed_out = [], [], False
    save(out / 'coordinate_checkpoint.json', best)
    for round_id in range(2):
        before = best['val_macro_f1']
        for node in order:
            incumbent = best
            node_best = best
            for wi in range(len(pool)):
                if wi == current[node]: continue
                candidate = current.copy(); candidate[node] = wi
                result = engine.evaluate(candidate, 'C_candidate', deadline)
                if result is None:
                    timed_out = True; break
                result.update(round=round_id+1, node=node)
                if result['val_macro_f1'] > node_best['val_macro_f1']: node_best = result
            if node_best['val_macro_f1'] > incumbent['val_macro_f1']:
                accepted.append(dict(round=round_id+1, node=node, old_sigma=float(pool[current[node]]),
                    new_sigma=node_best['widths'][node], old_f1=incumbent['val_macro_f1'], new_f1=node_best['val_macro_f1'],
                    evaluation_id=node_best['evaluation_id']))
                best = node_best; current = np.array(best['width_indices'])
                save(out / 'coordinate_checkpoint.json', best)
            if timed_out: break
        rounds.append(dict(round=round_id+1, start_f1=before, end_f1=best['val_macro_f1']))
        if timed_out or best['val_macro_f1'] == before: break
    count = sum(r['complete'] for r in engine.records[records0:])
    directed_info = dict(accepted=accepted, rounds=rounds, node_order=order, completed_configurations=count,
                         candidate_configurations_excluding_initial=count-1, ridge_fits=engine.fits-fits0,
                         seconds=time.perf_counter()-start, timed_out=timed_out)
    save(out / 'directed_search.json', directed_info)
    # Independent validation-selected random search, configuration-count matched.
    rng = np.random.default_rng(10042); matched = None
    start = time.perf_counter()
    for _ in range(count):
        result = engine.evaluate(rng.integers(0, len(pool), size=engine.m), 'E_matched_candidate')
        if matched is None or result['val_macro_f1'] > matched['val_macro_f1']: matched = result
    save(out / 'matched_budget.json', dict(configurations=count, ridge_fits=count*len(alphas), seconds=time.perf_counter()-start))
    final = {'C_directed': best, **{f'D_random_{i}': c for i, c in enumerate(random_configs)}, 'E_budget_matched': matched}
    for name, c in final.items():
        joblib.dump(dict(widths=c['widths'], width_indices=c['width_indices'], ridge=engine.fit_final(c),
                         centers=old['kmeans'].cluster_centers_, alpha=c['alpha']), out / (name+'.joblib'))
    save(out / 'all_validation_evaluations.json', engine.records)
    pd.DataFrame([dict(seed=c['seed'], val_macro_f1=c['val_macro_f1'], alpha=c['alpha'], seconds=c['elapsed_seconds']) for c in random_configs]).to_csv(out / 'random10_validation.csv', index=False)
    vals = np.array([c['val_macro_f1'] for c in random_configs])
    save(out / 'random10_validation_summary.json', dict(mean=float(vals.mean()), std_sample_ddof1=float(vals.std(ddof=1))))
    # All selections and all trained models persisted before test access.
    save(out / 'locked_configs.json', dict(locked_at_utc=datetime.now(timezone.utc).isoformat(), final=final,
         model_hashes={name: sha(out/(name+'.joblib')) for name in final}, protocol_sha256=sha(out/'protocol.json'),
         validation_records_sha256=sha(out/'all_validation_evaluations.json')))
    print('VALIDATION LOCKED', {n: c['val_macro_f1'] for n,c in final.items()}, flush=True)
    return final, directed_info


def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y, pred)), macro_f1=score(y, pred),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                per_class=classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0),
                confusion_counts=confusion_matrix(y, pred, labels=CLASSES).tolist(),
                confusion_normalized_true=confusion_matrix(y, pred, labels=CLASSES, normalize='true').tolist())


def cm_plot(met, name, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, fmt in zip(axes, ['confusion_counts', 'confusion_normalized_true'], ['d', '.2f']):
        cm = np.array(met[key]); ax.imshow(cm, cmap='Blues')
        for i in range(4):
            for j in range(4): ax.text(j, i, format(cm[i,j], fmt), ha='center', va='center')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='True', title=key)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out/(name+'_confusion.png'), dpi=150); plt.close(fig)


def preview(rows, path):
    n = max(len(rows), 1); canvas = Image.new('RGB', (4*260, ((n+3)//4)*245), 'white')
    draw = ImageDraw.Draw(canvas)
    if not len(rows): draw.text((15,15), 'No samples in this category', fill='black')
    for i, (_, row) in enumerate(rows.iterrows()):
        im = ImageOps.contain(Image.open(row.path).convert('RGB'), (250,185))
        x,y=(i%4)*260,(i//4)*245; canvas.paste(im,(x+(260-im.width)//2,y))
        draw.text((x+4,y+187),f'True: {row.true_label}\nLinear: {row.linear}  RBF: {row.rbf}\n{row.image_id[:24]}',fill='black')
    canvas.save(path)


def diagnostics_and_test(base, out, manifest, features, prep, final):
    assert (out/'locked_configs.json').is_file()
    lock = read(out/'locked_configs.json')
    assert all(sha(out/(name+'.joblib')) == lock['model_hashes'][name] for name in final)
    test = manifest.split.eq('test').to_numpy()
    test_rows = manifest.loc[test].copy()
    y = test_rows['class'].to_numpy()
    # First access to cached test targets in this process, after lock.
    with np.load(base/'features_resnet18.npz') as cached:
        assert np.array_equal(cached['labels'], manifest['class'].to_numpy())
    all_metrics = {}
    preds = test_rows[['sample_id','image_id','path']].copy()
    for name, stem in [('A_linear','linear'),('B_single','rbf')]:
        saved = pd.read_csv(base/(stem+'_test_predictions.csv')).set_index('sample_id').loc[test_rows.sample_id]
        assert np.array_equal(saved.true_label, y)
        met=metrics(y,saved.predicted_label)
        old=read(base/(stem+'_test_metrics.json'))
        assert all(np.isclose(met[k],old[k]) for k in METRICS)
        all_metrics[name]=met; preds[stem]=saved.predicted_label.to_numpy()
        save(out/(name+'_metrics.json'),met); cm_plot(met,name,out)
    preds['true_label']=y
    for name, mask in [('linear_correct_rbf_wrong', (preds.linear==y)&(preds.rbf!=y)),
                       ('rbf_correct_linear_wrong', (preds.rbf==y)&(preds.linear!=y))]:
        rows=preds.loc[mask]; rows.to_csv(out/(name+'.csv'),index=False); preview(rows,out/(name+'.jpg'))
    ztest=prep['pca'].transform(prep['scaler'].transform(features[test]))
    for name, config in final.items():
        model=joblib.load(out/(name+'.joblib')); xt=phi(ztest,model['centers'],model['widths'])
        pred=model['ridge'].predict(xt); met=metrics(y,pred); all_metrics[name]=met
        save(out/(name+'_metrics.json'),met); cm_plot(met,name,out)
        df=pd.DataFrame({'sample_id':test_rows.sample_id.to_numpy(),'true_label':y,'predicted_label':pred})
        scores=model['ridge'].decision_function(xt)
        for j,c in enumerate(model['ridge'].classes_): df['class_score_'+c]=scores[:,j]
        df.to_csv(out/(name+'_predictions.csv'),index=False)
    original=read(base/'validation_search.json')
    rows=[]; per_class=[]
    for name, met in all_metrics.items():
        val=final[name]['val_macro_f1'] if name in final else original['best_linear' if name=='A_linear' else 'best_rbf']['val_macro_f1']
        rows.append(dict(model=name,val_macro_f1=val,**{k:met[k] for k in METRICS}))
        for c in CLASSES: per_class.append(dict(model=name,label=c,**met['per_class'][c]))
    table=pd.DataFrame(rows); table.to_csv(out/'results.csv',index=False)
    pd.DataFrame(per_class).to_csv(out/'per_class_metrics.csv',index=False)
    random=table[table.model.str.startswith('D_random')]
    save(out/'random10_summary.json',{k:dict(mean=float(random[k].mean()),std_sample_ddof1=float(random[k].std(ddof=1))) for k in ['val_macro_f1']+METRICS})
    return table, all_metrics


def plots(out, original, final, directed):
    fig,ax=plt.subplots(figsize=(7,4)); scan=pd.DataFrame(original['rbf'])
    for alpha,rows in scan.groupby('alpha'):
        ax.plot(rows.sigma, rows.val_macro_f1, marker='o', label=f'alpha={alpha:g}')
    ax.set(xscale='log',xlabel='sigma',ylabel='Validation Macro-F1',title='Original single-width scan'); ax.legend(); fig.tight_layout(); fig.savefig(out/'single_sigma_scan.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); f=original['best_rbf']['val_macro_f1']; records=read(out/'all_validation_evaluations.json'); history=[]
    for r in records:
        if r['group'].startswith('C_') and r['complete']:
            f=max(f,r['val_macro_f1']); history.append(f)
    ax.plot(range(1,len(history)+1),history); ax.set(xlabel='Completed configuration evaluations',ylabel='Best validation Macro-F1',title='Directed coordinate search'); fig.tight_layout(); fig.savefig(out/'directed_search.png',dpi=160); plt.close(fig)
    pool=read(out/'protocol.json')['pool']
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for name in ['C_directed','E_budget_matched']:
        widths=final[name]['widths']; axes[0].plot(range(len(widths)),np.array(widths)/pool[2],'.-',label=name)
    axes[0].axhline(original['best_rbf']['sigma']/pool[2],ls='--',color='gray',label='B_single'); axes[0].set(xlabel='Node',ylabel='sigma / sigma_ref'); axes[0].legend()
    names=['B_single']+list(final)
    counts=[np.bincount([pool.index(original['best_rbf']['sigma'])]*32,minlength=5)]+[np.bincount(final[n]['width_indices'],minlength=5) for n in final]
    im=axes[1].imshow(np.array(counts).T,aspect='auto',cmap='Blues')
    axes[1].set(yticks=range(5),yticklabels=['0.25','0.5','1','2','4'],ylabel='sigma / sigma_ref',xticks=range(len(names)),xticklabels=names)
    axes[1].tick_params(axis='x',rotation=60); fig.colorbar(im,ax=axes[1],label='Node count'); fig.tight_layout(); fig.savefig(out/'node_width_distribution.png',dpi=160); plt.close(fig)


def reports(base, out):
    audit = read(out/'baseline_audit.json')
    table = pd.read_csv(out/'results.csv')
    stats = read(out/'random10_summary.json')
    directed = read(out/'directed_search.json')
    matched = read(out/'matched_budget.json')
    response = read(out/'train_rbf_responses.json')
    locked = read(out/'locked_configs.json')['final']
    perclass = pd.read_csv(out/'per_class_metrics.csv')
    counts = pd.read_csv(out/'sample_counts.csv')
    def md(df):
        headers = list(df.columns)
        def cell(x): return f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x)
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + ['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False, name=None)])
    summary = table[table.model.isin(['A_linear','B_single','C_directed','E_budget_matched'])].copy()
    random_line = {'model':'D_random_10_mean'}
    random_line.update({k:stats[k]['mean'] for k in ['val_macro_f1']+METRICS})
    summary = pd.concat([summary, pd.DataFrame([random_line])],ignore_index=True)
    summary.to_csv(out/'comparison_summary.csv',index=False)
    deltas = []
    b = perclass[perclass.model.eq('B_single')].set_index('label')
    for model in ['C_directed','E_budget_matched']:
        c = perclass[perclass.model.eq(model)].set_index('label')
        for label in CLASSES:
            deltas.append({'model':model,'class':label,'single_f1':b.loc[label,'f1-score'],
                           'new_f1':c.loc[label,'f1-score'],'delta_f1':c.loc[label,'f1-score']-b.loc[label,'f1-score']})
    delta=pd.DataFrame(deltas); delta.to_csv(out/'class_f1_changes.csv',index=False)
    changes=pd.DataFrame(directed['accepted']); changes.to_csv(out/'accepted_coordinate_changes.csv',index=False)
    widths=pd.DataFrame({'node':range(32),**{n:c['widths'] for n,c in locked.items()}})
    widths.to_csv(out/'final_width_vectors.csv',index=False)
    audit_text = f'''# 基线审计

用户给出的六个分数是测试集分数。源代码与已保存 JSON/CSV 一致；验证集分数分别为 Linear 0.926867、单宽度 RBF 0.926984。

数据版本：HaGRIDv2 512px 的第三方派生子集 zhouxzh/portable-hagridv2-mediapipe-hand，revision e343025bd4fd37719dbc269201d6b24124f0c432。该子集经 MediaPipe 质量筛选，存在样本选择偏差。

{md(counts)}

训练912、验证106、测试278；四类共1296个样本。三组 user_id 全部缺失，实际用户数及 train/val/test 用户交集均为未知，不能写作0个用户或0交集。图片ID无跨集合重复；已逐行核验1296条 manifest 与发布者 source_split、原图ID和原始框一致。第三方保留的 source_split 无法替代原始官方标注的独立用户核验。

编码器 torchvision ResNet18，权重 ResNet18_Weights.IMAGENET1K_V1，eval/无梯度，fc=Identity得到512维；确定性 Resize256 / CenterCrop224 / ImageNet归一化。PCA前 StandardScaler，训练集拟合，32维，whiten=False。KMeans为32中心、random_state=42、n_init=10、k-means++。

sigma_ref=20.39389991760254；宽度池=[5.098474979400635,10.19694995880127,20.39389991760254,40.78779983520508,81.57559967041016]。最佳单宽度=sigma_ref、alpha=0.01；Linear alpha=0.001。alpha候选为[0.001,0.01,0.1,1,10,100]，由验证Macro-F1选择。

缓存ID和split与manifest逐行一致；两基线源代码使用同一缓存和共享预处理。StandardScaler统计量匹配912个训练特征，PCA样本数及均值匹配训练集；KMeans训练标签长度为912，中心与训练簇均值最大误差为{audit['center_mean_max_error']:.8g}。未重拟合PCA或KMeans，保存模型在验证集的预测重新核验后与原记录相同。源代码显示测试未参与原超参数选择；历史运行过程未保存隔离追踪，不能据此证明不存在未记录的人为测试反馈。

修复前：原脚本按Parquet文件名分配split，未逐行验证source_split。修复后：新脚本增加逐行核验、缓存对应检查、训练统计量检查、配置锁定和基线全文件SHA-256保护，全部通过。原代码、模型、特征、划分和结果保持原样，未发现需要改动基线的统计量不一致。user_id缺失无法在当前数据内修复。

另一个限制：原编码器中心裁剪是否保留所有关键手指未做定量审计；本轮按照冻结要求沿用原缓存。这不影响本轮配置间的共享输入，但限制整体实验质量结论。
'''
    (out/'BASELINE_AUDIT_ZH.md').write_text(audit_text,encoding='utf-8')
    report = f'''# 多宽度 RBF 对照实验

HaGRID小规模子集、真实框裁剪、冻结视觉编码器的初步分类实验。本轮是同一保留测试集上的探索性软件对照，不是新的独立确认实验。缺少user_id，只能维持真实数据流程调试结论，尚未验证跨用户泛化或硬件优势。

## 结果

{md(summary)}

定向多宽度验证Macro-F1由0.926984升至0.961903（+0.034918），测试Macro-F1由0.930436降至0.916340（-0.014096）。预算匹配随机搜索验证更高（0.965638），测试0.922045仍低于单宽度。不能据此声称多宽度提高测试性能。

定向多宽度测试Macro-F1高于10随机配置均值0.911822，但低于单宽度；这种比较不足以证明验证引导选择普遍优越，尤其定向搜索使用更多验证反馈。以下列出所有随机种子，未按测试分数筛选或删除配置。

{md(table[table.model.str.startswith('D_random')])}

随机组均值±样本标准差（ddof=1）：验证Macro-F1 {stats['val_macro_f1']['mean']:.6f}±{stats['val_macro_f1']['std_sample_ddof1']:.6f}；测试Accuracy {stats['accuracy']['mean']:.6f}±{stats['accuracy']['std_sample_ddof1']:.6f}；测试Macro-F1 {stats['macro_f1']['mean']:.6f}±{stats['macro_f1']['std_sample_ddof1']:.6f}；测试Balanced Accuracy {stats['balanced_accuracy']['mean']:.6f}±{stats['balanced_accuracy']['std_sample_ddof1']:.6f}。随机种子标准差反映宽度随机配置的变化，不等于测试总体不确定性或置信区间。

## 每类变化（对照单宽度）

{md(delta)}

定向多宽度fist F1持平、like略升、peace与ok下降。peace召回仍为50/57，但精确率从0.980392降至0.892857；ok召回从99/100降至95/100。观察与验证集适应过强的可能性相容，但本实验没有证明唯一因果原因。验证集只有106个样本，其中fist仅8个，反复验证反馈可能导致选择偏差。

## 基线诊断

直接读取原预测，Linear正确而RBF错误的样本4个，反向1个，列表与全部5张样本预览已保存。每类Precision/Recall/F1/support及原始/归一化混淆矩阵覆盖所有模型。

训练集最佳单宽度响应：最小0.141608，1%分位0.294676，中位0.606531，99%分位0.868078，最大0.961504。节点标准差范围{min(response['node_std']):.6f}至{max(response['node_std']):.6f}。按<1e-6或>1-1e-6定义的近0/近1响应均为0%，且所有值有限、无全零行。未见明显饱和；这不能证明节点表达能力充分，也不能把响应分布认定为误分类的原因。

## 搜索设置与预算

固定原512维缓存、StandardScaler、32维PCA、32个KMeans中心与数据划分。三种RBF始终32节点、32维输出，每节点仅一个宽度，未拼接各宽度。统一使用原5宽度池和6个alpha。

10随机配置：种子0至9，NumPy default_rng独立均匀抽取32个池索引，每配置6次训练Ridge/验证选alpha，共60次搜索拟合。各次耗时、alpha、完整宽度向量已保存。

定向配置：种子42固定节点顺序、最多2轮，尝试每节点的其他4宽度，选严格改善的候选；并列保留当前配置。接受4次修改，第二轮无改善停止。共{directed['completed_configurations']}个完整配置（含1次初始化，256次修改候选）、{directed['ridge_fits']}次搜索Ridge拟合，耗时{directed['seconds']:.3f}秒，未触发900秒上限。最终alpha=0.1，节点26/27/31宽度为0.5×sigma_ref、节点14为2×sigma_ref，其余28节点为sigma_ref，节点编号从0开始。

预算匹配随机搜索：单独seed=10042，{matched['configurations']}个随机配置，{matched['ridge_fits']}次搜索Ridge拟合，耗时{matched['seconds']:.3f}秒；只按验证分数选最优，最终alpha=0.001。与定向搜索匹配配置评估次数及alpha拟合次数，并不保证信息反馈或其他计算开销完全相同。该最佳随机配置与10随机配置均值分开报告。各最终新模型另做1次训练拟合以落盘，共12次，未计入搜索拟合预算；不使用验证样本训练。

所有新模型在locked_configs.json写入后才访问缓存测试标签和已有测试预测；代码保存锁定配置、验证日志及模型哈希。基线测试不重跑；新模型各统一评估一次。测试结果没有反馈到搜索。

## 产物和复现

见README.md。保留原实验全部文件SHA-256，核验原文件未修改。建议后续先补齐官方user_id并扩大独立验证数据，再进行新的预先规定对照；本轮已停止搜索，不作硬件或摄像头集成。
'''
    (out/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    (out/'README.md').write_text('''# Reproduce this experiment

From /home/ziyan/FYP_2026, reuse the existing .venv:

```bash
.venv/bin/python multiwidth_experiment.py --base artifacts/hagrid_small --out artifacts/hagrid_multiwidth_reproduction
```

The output directory must not already exist. No downloads, encoder inference, preprocessing refit or new KMeans fit occur. The baseline directory is read-only to this script. The full protocol is saved before search, and configuration/model hashes are locked before new test evaluation. BLAS threads are limited to one; timings are machine-dependent. The source archive and source Parquet files referenced by the baseline manifest must remain available for audit.

- BASELINE_AUDIT_ZH.md / baseline_audit.json: evidence and remaining identity limitation.
- results.csv / comparison_summary.csv / per_class_metrics.csv: complete outcomes.
- all_validation_evaluations.json: every alpha for every evaluated configuration.
- directed_search.json / accepted_coordinate_changes.csv: moves, counts, time.
- random10_validation.csv / random10_summary.json / matched_budget.json: random results and budgets.
- protocol.json / locked_configs.json / final_width_vectors.csv: frozen selections.
- *.joblib: final models, widths and original centers; baseline_preprocessing.joblib: copied scaler/PCA.
- *_predictions.csv: test predictions and class scores (not probabilities).
- *_confusion.png: count and true-normalized matrices, also in *_metrics.json.
- linear_correct_rbf_wrong.csv/.jpg and rbf_correct_linear_wrong.csv/.jpg: all disagreement cases.
- single_sigma_scan.png / directed_search.png / node_width_distribution.png: figures.
- baseline_hashes_before.json / verification.json: preservation checks.

The original features_resnet18.npz is reused in place. User identities are missing, so no verified cross-user claim is permitted. This is an exploratory comparison on an already used holdout, not independent confirmation.
''',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,default=Path('artifacts/hagrid_small')); ap.add_argument('--out',type=Path,default=Path('artifacts/hagrid_multiwidth_v1')); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    before=snapshot(args.base); save(args.out/'baseline_hashes_before.json',before)
    shutil.copy2(__file__,args.out/'multiwidth_experiment.py')
    with threadpool_limits(limits=1):
        manifest,features,ztr,zval,ytr,yval,old,original,prep=audit(args.base,args.out)
        print('AUDIT PASSED; user identity remains unavailable.',flush=True)
        final,directed=select(args.out,ztr,zval,ytr,yval,old,original)
        table,mets=diagnostics_and_test(args.base,args.out,manifest,features,prep,final)
        plots(args.out,original,final,directed)
    after=snapshot(args.base); assert before==after
    save(args.out/'verification.json',dict(baseline_unchanged=True, files_checked=len(before),
        all_new_rbf_dimensions=32, final_models=len(final), all_selections_locked_before_test=True,
        test_evaluation='Exploratory same held-out set', user_disjointness='UNKNOWN'))
    reports(args.base, args.out)
    print(table.to_string(index=False)); print('DIRECTED',directed); print('RANDOM SUMMARY',read(args.out/'random10_summary.json'))


if __name__=='__main__': main()
