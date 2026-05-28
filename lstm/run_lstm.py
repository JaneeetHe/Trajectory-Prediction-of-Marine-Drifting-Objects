"""Train 3 LSTMs (vanilla / phys-residual / phys-residual+loss) x 3 seeds"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm as L
from metrics import evaluate


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASELINE_NPZ = REPO / 'baselines' / 'baseline_predictions.npz'
BASELINE_CSV = REPO / 'baselines' / 'baseline_metrics.csv'

SEEDS = [7, 42, 123]
CANONICAL_SEED = 42

MODELS = [
    ('vanilla',   'LSTM (vanilla)',            False, 0.0),
    ('resid',     'LSTM (phys-residual)',      True,  0.0),
    ('residloss', 'LSTM (phys-residual+loss)', True,  1.0),
]


def main():
    data = L.load_data()
    base = np.load(BASELINE_NPZ, allow_pickle=True)
    assert np.allclose(data['y_test'], base['y_test']) # test on same data
    assert np.allclose(data['cur_lat_test'], base['cur_lat_test'])
    print(f"train {data['y_train'].shape}  val {data['y_val'].shape}  test {data['y_test'].shape}")

    d = L.prepare_splits(data)
    del data

    # advection prior should match baseline ~8.721 km
    r_adv, _ = evaluate(d['yte'], d['adv_te'], d['cur_lat_te'], d['cur_lon_te'], 'adv')
    print(f"advection check: mean_geo = {r_adv['mean_geo']:.4f} km")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = {}
    preds_per_seed, geos_per_seed = {}, {}

    for key, label, physics, lam in MODELS:
        print(f"\n=== {label} ===")
        per_seed, preds, geos = [], [], []
        for seed in SEEDS:
            cfg = dict(seed=seed, physics_informed=physics, lambda_phys=lam)
            model, rec = L.train_model(d, cfg, label=f"{label}/s{seed}")
            y_pred = L.predict(model, d['Xte'], d['adv_te'], physics, device)
            res, geo = evaluate(d['yte'], y_pred, d['cur_lat_te'], d['cur_lon_te'], label)
            rec['seed'] = seed
            rec['test_metrics'] = {k: float(v) for k, v in res.items() if k != 'label'}
            per_seed.append(rec)
            preds.append(y_pred)
            geos.append(geo)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        agg = {} # aggregate over seeds to get mean ± range for each metric
        for k in ('mae_lat', 'mae_lon', 'rmse_lat', 'rmse_lon', 'mean_geo', 'median_geo', 'p90_geo'):
            vals = np.array([r['test_metrics'][k] for r in per_seed])
            agg[k] = {'mean': float(vals.mean()), 'min': float(vals.min()),
                      'max': float(vals.max()), 'range': float(vals.max() - vals.min())}
        records[key] = {'label': label,
                        'cfg_common': dict(physics_informed=physics, lambda_phys=lam),
                        'per_seed': per_seed, 'aggregated_test': agg}
        preds_per_seed[key] = np.stack(preds)
        geos_per_seed[key] = np.stack(geos)
        print(f"  -> {label}: mean_geo = {agg['mean_geo']['mean']:.3f} ± {agg['mean_geo']['range']:.3f} km")

    # combined metrics: baselines + LSTMs with mean-over-seed range
    with open(BASELINE_CSV, encoding='utf-8') as f:
        base_rows = [r for r in csv.reader(f) if r and r[0].strip()]
    header = base_rows[0][:6] + ['Mean geo range (km)'] + base_rows[0][6:]
    base_padded = [r[:6] + [''] + r[6:] for r in base_rows[1:]]
    my_rows = []
    for key, label, _, _ in MODELS:
        a = records[key]['aggregated_test']
        my_rows.append([label,
                        f"{a['mae_lat']['mean']:.5f}", f"{a['mae_lon']['mean']:.5f}",
                        f"{a['rmse_lat']['mean']:.5f}", f"{a['rmse_lon']['mean']:.5f}",
                        f"{a['mean_geo']['mean']:.3f}", f"{a['mean_geo']['range']:.3f}",
                        f"{a['median_geo']['mean']:.3f}", f"{a['p90_geo']['mean']:.3f}"])

    with open(HERE / 'lstm_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(base_padded + my_rows)

    canon = SEEDS.index(CANONICAL_SEED)
    out = dict(
        y_test=d['yte'], cur_lat_test=d['cur_lat_te'], cur_lon_test=d['cur_lon_te'],
        adv_test=d['adv_te'],
        seeds=np.array(SEEDS, dtype=np.int64),
        canonical_seed=np.array(CANONICAL_SEED, dtype=np.int64),
        model_labels=np.array([label for _, label, _, _ in MODELS]),
        model_keys=np.array([key for key, _, _, _ in MODELS]),
    )
    for key, _, _, _ in MODELS:
        out[f'y_lstm_{key}']         = preds_per_seed[key][canon]
        out[f'geo_lstm_{key}']       = geos_per_seed[key][canon]
        out[f'y_lstm_{key}_seeds']   = preds_per_seed[key]
        out[f'geo_lstm_{key}_seeds'] = geos_per_seed[key]
    np.savez_compressed(HERE / 'lstm_predictions.npz', **out)

    with open(HERE / 'lstm_run.json', 'w') as f:
        json.dump({'data_root': str(L.DATA_ROOT),
                   'lambda_phys': 1.0,
                   'seeds': SEEDS, 'canonical_seed': CANONICAL_SEED,
                   'use_gpu_resident': True,
                   'models': records},
                  f, indent=2, default=float)

    print()
    for r in [header] + base_padded + my_rows:
        print("  ".join(r))


if __name__ == '__main__':
    main()