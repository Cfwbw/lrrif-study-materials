from pathlib import Path
from itertools import combinations
import argparse
import csv
import hashlib
import json
import numpy as np

B = 10000
SEED = 20260923
RATERS = ['rater_01', 'rater_02', 'rater_03']
DIMS = [f'D{i}' for i in range(1, 9)]
MODELS = [
    ('siliconflow_minimax_m2_5_paid', 'MiniMax-M2.5'),
    ('openrouter_gpt_oss_20b_free', 'GPT-OSS-20B'),
    ('siliconflow_deepseek_v4_flash_paid', 'DeepSeek-V4-Flash'),
    ('siliconflow_primary', 'Qwen3-8B baseline'),
]
METRICS = ['ICC_2_1', 'ICC_2_3', 'exact_agreement', 'within_one_agreement',
           'kappa_01_02', 'kappa_01_03', 'kappa_02_03', 'alpha_ordinal']
PAIRS = list(combinations(range(3), 2))


def load_ratings(path):
    """Validate the 4,800 public rating rows and reshape to output × rater × dimension."""
    meta_fields = ['output_id', 'case_id', 'participant_id', 'date', 'arm', 'model_key']
    required = ['scoring_id', 'rater_id', *meta_fields, *DIMS]
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing rating columns: {sorted(missing)}')
        rows = list(reader)
    if len(rows) != 4800 or {r['rater_id'] for r in rows} != set(RATERS):
        raise ValueError('Expected 4,800 rows scored by rater_01, rater_02 and rater_03.')
    by_id = {}
    for row in rows:
        if any(not row[field].strip() for field in required):
            raise ValueError('Required rating identifiers, metadata and scores must be nonempty.')
        sid = row['scoring_id']
        previous = by_id.setdefault(sid, row)
        if any(row[field] != previous[field] for field in meta_fields):
            raise ValueError(f'Inconsistent metadata across raters for {sid}.')
    ids = sorted(by_id)
    meta = [{field: by_id[sid][field] for field in ['scoring_id', *meta_fields]} for sid in ids]
    if len(ids) != 1600 or len({r['output_id'] for r in meta}) != 1600:
        raise ValueError('Expected 1,600 scoring IDs in one-to-one correspondence with outputs.')
    cases = {}
    for row in meta:
        key = (row['participant_id'], row['date'])
        if cases.setdefault(row['case_id'], key) != key:
            raise ValueError(f'Case {row["case_id"]} has inconsistent participant/date metadata.')
    if len(cases) != 100 or len(set(cases.values())) != 100:
        raise ValueError('Expected 100 distinct person-day cases.')
    if len({r['participant_id'] for r in meta}) != 16:
        raise ValueError('Expected 16 participants.')
    expected = {(case, model, arm) for case in cases for model, _ in MODELS for arm in 'ABCD'}
    observed = {(r['case_id'], r['model_key'], r['arm']) for r in meta}
    if observed != expected:
        raise ValueError('Outputs must cover all 100 cases × four endpoints × four prompt arms.')
    imap = {sid: i for i, sid in enumerate(ids)}
    values = np.full((1600, 3, 8), np.nan)
    for row in rows:
        ix = (imap[row['scoring_id']], RATERS.index(row['rater_id']))
        if np.isfinite(values[ix]).any():
            raise ValueError(f'Duplicate rating for {row["scoring_id"]}/{row["rater_id"]}.')
        scores = np.array([float(row[dim]) for dim in DIMS])
        if not np.isin(scores, [1, 2, 3, 4, 5]).all():
            raise ValueError('All dimension scores must be integers from 1 to 5.')
        values[ix] = scores
    if not np.isfinite(values).all():
        raise ValueError('Each output must have all eight scores from all three raters.')
    return values, meta


def create_bootstrap_design(meta, out):
    """Save shared participant-cluster bootstrap multiplicities."""
    participants = sorted({row['participant_id'] for row in meta})
    if len(participants) != 16:
        raise ValueError('Expected 16 participant clusters.')
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(participants), size=(B, len(participants)))
    weights = np.zeros((B, len(participants)), dtype=np.int16)
    for i in range(B):
        weights[i] = np.bincount(draws[i], minlength=len(participants))
    out.mkdir(parents=True, exist_ok=True)
    weight_path = out / 'participant_weights.npy'
    np.save(weight_path, weights, allow_pickle=False)
    (out / 'participant_levels.json').write_text(json.dumps(participants, indent=2) + '\n')
    manifest = {
        'replicates': B, 'rng': 'numpy.random.Generator(PCG64)', 'numpy': np.__version__,
        'participant': {'seed': SEED, 'levels': participants, 'path': str(weight_path),
                        'shape': list(weights.shape),
                        'sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest()},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def load_bootstrap_design(meta, design):
    participants = json.loads((design / 'participant_levels.json').read_text())
    manifest = json.loads((design / 'manifest.json').read_text())
    path = design / 'participant_weights.npy'
    weights = np.load(path, allow_pickle=False)
    expected = sorted({row['participant_id'] for row in meta})
    if participants != expected or manifest['participant']['levels'] != expected:
        raise ValueError('Bootstrap participants do not match the rating data.')
    if manifest['replicates'] != B or manifest['participant']['seed'] != SEED:
        raise ValueError('Bootstrap design must use 10,000 replicates and seed 20260923.')
    if weights.shape != (B, 16) or not np.issubdtype(weights.dtype, np.integer):
        raise ValueError('Expected a 10,000 × 16 integer bootstrap-weight matrix.')
    if np.any(weights < 0) or not np.all(weights.sum(axis=1) == 16):
        raise ValueError('Every bootstrap replicate must sample 16 participant clusters.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['participant']['sha256']:
        raise ValueError('Bootstrap weights do not match their recorded checksum.')
    return participants, weights.astype(float)


def strata(meta):
    all_ix = np.arange(len(meta))
    groups = [('overall','all','All outputs','','',all_ix)]
    for model,label in MODELS:
        groups.append(('endpoint', model, label, model, '',
                       np.array([i for i,r in enumerate(meta) if r['model_key']==model])))
    for arm in 'ABCD':
        groups.append(('arm',arm,'Arm '+arm,'',arm,
                       np.array([i for i,r in enumerate(meta) if r['arm']==arm])))
    return groups

def sufficient_statistics(values, cluster_indices, n_clusters):
    """values: output by 3 rater; result: cluster by 45 sufficient totals."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    count = np.stack([(x==i).sum(axis=1) for i in range(1,6)], axis=1).astype(float)
    coincidence = count[:,:,None] * count[:,None,:]
    for j in range(5):
        coincidence[:,j,j] -= count[:,j]
    coincidence /= 2.0
    features = np.column_stack([
        np.ones(n), x.sum(axis=1), (x*x).sum(axis=1), x.sum(axis=1)**2,
        x, x*x,
        np.ptp(x,axis=1)==0, np.ptp(x,axis=1)<=1,
        *[(x[:,a]-x[:,b])**2 for a,b in PAIRS],
        count, coincidence.reshape(n,25),
    ])
    # n,T,sum(x^2),sum(row_total^2), rater sums(3), rater squares(3),
    # exact,within, pair squared-difference sums(3), category counts(5), coincidences(25).
    assert features.shape[1] == 45
    out = np.zeros((n_clusters, features.shape[1]))
    np.add.at(out, cluster_indices, features)
    return out

def safe_divide(num, den):
    result = np.full(np.broadcast_shapes(np.shape(num),np.shape(den)), np.nan)
    np.divide(num, den, out=result, where=np.abs(den)>1e-12)
    return result

def metrics_from_totals(s):
    """Compute metrics from weighted totals; rows are bootstrap replicates."""
    s = np.atleast_2d(s)
    n,t,ss,rowss = s[:,0],s[:,1],s[:,2],s[:,3]
    cols, colsq = s[:,4:7],s[:,7:10]
    ms_output = safe_divide(rowss/3 - t*t/(n*3),n-1)
    ms_rater = (np.sum(cols*cols,axis=1)/n - t*t/(n*3))/2
    sse = ss - rowss/3 - np.sum(cols*cols,axis=1)/n + t*t/(n*3)
    if np.any(sse < -1e-8):
        raise ValueError('Negative error sum of squares beyond roundoff')
    sse = np.maximum(sse,0)
    ms_error = safe_divide(sse,(n-1)*2)
    numerator = ms_output-ms_error
    icc1 = safe_divide(numerator,ms_output+2*ms_error+3*(ms_rater-ms_error)/n)
    icc3 = safe_divide(numerator,ms_output+(ms_rater-ms_error)/n)
    kappas = []
    for p,(a,b) in enumerate(PAIRS):
        # Algebraically cancel n^2 before dividing; integer-scale sums avoid
        # cancellation around kappa=0 when one rater is constant.
        observed = s[:,12+p]*n
        expected = (colsq[:,a]+colsq[:,b])*n - 2*cols[:,a]*cols[:,b]
        kappas.append(1-safe_divide(observed,expected))
    kap = np.column_stack(kappas)
    marginal = s[:,15:20]
    midrank = np.cumsum(marginal,axis=1) - marginal/2
    distance = (midrank[:,:,None]-midrank[:,None,:])**2
    coincidence = s[:,20:45].reshape(-1,5,5)
    observed = np.sum(coincidence*distance,axis=(1,2))
    expected = np.sum(marginal[:,:,None]*marginal[:,None,:]*distance,axis=(1,2))/(3*n-1)
    alpha = 1-safe_divide(observed,expected)
    return np.column_stack([icc1,icc3,s[:,10]/n,s[:,11]/n,kap,alpha])

def formatted(v,lo,hi):
    if not np.isfinite(v):
        return 'NA'
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f'{v:.3f} [NA]'
    return f'{v:.3f} [{lo:.3f}, {hi:.3f}]'

def run_reliability(data, meta, out, design):
    participants, weights = load_bootstrap_design(meta, design)
    participant_ix = np.array([participants.index(row['participant_id']) for row in meta])
    rows, wide_rows = [], []
    for stype, sid, label, model, arm, indices in strata(meta):
        for d, dim in enumerate(DIMS):
            sufficient = sufficient_statistics(data[indices, :, d], participant_ix[indices], 16)
            point = metrics_from_totals(sufficient.sum(axis=0))[0]
            boot = metrics_from_totals(weights @ sufficient)
            base = {
                'stratum_type': stype, 'stratum_id': sid, 'stratum_label': label,
                'model_key': model, 'arm': arm, 'dimension': dim,
                'n_outputs': len(indices), 'n_cases': len({meta[i]['case_id'] for i in indices}),
                'n_participants': len(set(participant_ix[indices])), 'n_raters': 3,
                'ci_method': 'participant_cluster_percentile', 'bootstrap_replicates': B,
                'seed': SEED, 'rater_scope': 'conditional_on_observed_three_experts',
            }
            wide = dict(base)
            for m, metric in enumerate(METRICS):
                finite = np.isfinite(boot[:, m])
                nv = int(finite.sum())
                # Undefined replicates are disclosed, never replaced or clipped.
                lo, hi = np.quantile(boot[finite, m], [.025, .975], method='linear') if nv else [np.nan, np.nan]
                status = ('undefined_point' if not np.isfinite(point[m]) else
                          'all_replicates_undefined' if not nv else
                          'partial_undefined_replicates' if nv < B else
                          'degenerate_empirical_interval' if abs(hi-lo) < 1e-14 else 'ok')
                rows.append({**base, 'metric': metric, 'estimate': point[m],
                             'ci95_low': lo, 'ci95_high': hi, 'n_valid_bootstrap': nv,
                             'n_undefined_bootstrap': B-nv, 'interval_status': status})
                wide[metric+'_estimate'] = point[m]
                wide[metric+'_ci95_low'] = lo
                wide[metric+'_ci95_high'] = hi
                wide[metric+'_n_valid'] = nv
                wide[metric+'_status'] = status
                suffix = ' †' if nv < B else (' ‡' if status == 'degenerate_empirical_interval' else '')
                wide[metric+'_formatted'] = formatted(point[m], lo, hi) + suffix
            wide_rows.append(wide)
    out.mkdir(parents=True, exist_ok=True)
    tables = {'reliability_all_metrics_long.csv': rows,
              'reliability_all_strata_wide.csv': wide_rows,
              'reliability_pairwise_kappa.csv': [row for row in rows if row['metric'] in METRICS[4:7]]}
    for stype in ['overall', 'endpoint', 'arm']:
        tables[f'reliability_{stype}.csv'] = [row for row in wide_rows if row['stratum_type'] == stype]
    for name, table in tables.items():
        with (out / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ratings', type=Path,
                        default=Path(__file__).resolve().parent.parent / '04_ratings/ratings_4800.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-dir', type=Path,
                        help='Shared bootstrap directory; defaults to OUTPUT_DIR/bootstrap.')
    parser.add_argument('--mode', choices=['bootstrap', 'reliability', 'all'], default='all')
    args = parser.parse_args(argv)
    data, meta = load_ratings(args.ratings.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    design = args.bootstrap_dir.expanduser().resolve() if args.bootstrap_dir else output / 'bootstrap'
    if args.mode in ['bootstrap', 'all']:
        create_bootstrap_design(meta, design)
    if args.mode in ['reliability', 'all']:
        run_reliability(data, meta, output / 'reliability', design)


if __name__ == '__main__':
    main()
