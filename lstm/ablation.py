"""Ablation study""" 
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm as L
from metrics import evaluate


HERE = Path(__file__).resolve().parent


def build_configs():
    runs = []
    # model capacity (3 x 3)
    for h in [64, 128, 256]:
        for l in [1, 2, 3]:
            runs.append(dict(tag=f"A_cap_h{h}_l{l}", axis="capacity",
                             hidden_dim=h, num_layers=l, dropout=0.1,
                             lambda_phys=1.0, seed=42, features="all21"))
    # dropout (0.1 already in capacity)
    for dr in [0.0, 0.2]:
        runs.append(dict(tag=f"B_dropout_dr{dr}", axis="dropout",
                         hidden_dim=128, num_layers=2, dropout=dr,
                         lambda_phys=1.0, seed=42, features="all21"))
    # lambda_phys (1.0 already in capacity)
    for lam in [0.0, 0.01, 0.1]:
        runs.append(dict(tag=f"C_lambda_lam{lam}", axis="lambda_phys",
                         hidden_dim=128, num_layers=2, dropout=0.1,
                         lambda_phys=lam, seed=42, features="all21"))
    # feature subset
    runs.append(dict(tag="D_core4", axis="feature_subset",
                     hidden_dim=128, num_layers=2, dropout=0.1,
                     lambda_phys=1.0, seed=42, features="core4"))
    # seed stability (42 already in capacity)
    for s in [7, 123]:
        runs.append(dict(tag=f"E_seed_s{s}", axis="seed",
                         hidden_dim=128, num_layers=2, dropout=0.1,
                         lambda_phys=1.0, seed=s, features="all21"))
    return runs


def main():
    data = L.load_data()
    fi = data['fi']
    d = L.prepare_splits(data)
    del data
    core4_idx = [fi['latitude'], fi['longitude'], fi['ve'], fi['vn']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = build_configs()
    results = []
    t0 = time.time()

    for i, c in enumerate(configs, 1):
        print(f"\n[{i:02d}/{len(configs)}] {c['tag']} ({c['axis']})")
        if c['features'] == 'core4':
            d_use = dict(d,
                         Xtr=np.ascontiguousarray(d['Xtr'][:, :, core4_idx]),
                         Xva=np.ascontiguousarray(d['Xva'][:, :, core4_idx]),
                         Xte=np.ascontiguousarray(d['Xte'][:, :, core4_idx]))
            input_dim = 4
        else:
            d_use = d
            input_dim = 21

        cfg = dict(input_dim=input_dim,
                   hidden_dim=c['hidden_dim'], num_layers=c['num_layers'], dropout=c['dropout'],
                   lr=1e-3, batch_size=1024, epochs=25, patience=5,
                   physics_informed=True, lambda_phys=c['lambda_phys'], seed=c['seed'])
        t = time.time()
        model, rec = L.train_model(d_use, cfg, label=c['tag'])
        y_te = L.predict(model, d_use['Xte'], d_use['adv_te'], True, device)
        res, _ = evaluate(d_use['yte'], y_te, d_use['cur_lat_te'], d_use['cur_lon_te'], c['tag'])
        wall = time.time() - t

        results.append(dict(
            tag=c['tag'], axis=c['axis'],
            hidden_dim=c['hidden_dim'], num_layers=c['num_layers'],
            dropout=c['dropout'], lambda_phys=c['lambda_phys'],
            input_dim=input_dim, seed=c['seed'],
            feature_subset=c['features'],
            best_val_mean_geo=rec['best_val_mean_geo'], best_epoch=rec['best_epoch'],
            test_mean_geo=float(res['mean_geo']),
            test_median_geo=float(res['median_geo']),
            test_p90_geo=float(res['p90_geo']),
            n_epochs=len(rec['history']), wall_seconds=wall,
        ))
        print(f"  val={rec['best_val_mean_geo']:.4f}  test={res['mean_geo']:.4f}  wall={wall:.0f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r['best_val_mean_geo'])
    elapsed = time.time() - t0

    with open(HERE / 'ablation_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank_by_val', 'tag', 'axis', 'hidden', 'layers', 'dropout',
                    'lambda_phys', 'features', 'seed',
                    'val_mean_geo', 'test_mean_geo', 'test_median', 'test_p90',
                    'epochs', 'wall_s'])
        for rank, r in enumerate(results, 1):
            w.writerow([rank, r['tag'], r['axis'], r['hidden_dim'], r['num_layers'],
                        r['dropout'], r['lambda_phys'], r['feature_subset'], r['seed'],
                        f"{r['best_val_mean_geo']:.4f}", f"{r['test_mean_geo']:.4f}",
                        f"{r['test_median_geo']:.4f}", f"{r['test_p90_geo']:.4f}",
                        r['n_epochs'], f"{r['wall_seconds']:.1f}"])

    with open(HERE / 'ablation_results.json', 'w') as f:
        json.dump({'elapsed_sec': elapsed,
                   'n_runs': len(results),
                   'best_by_val': results[0]['tag'],
                   'records': results}, f, indent=2, default=float)

    print(f"\nablation done in {elapsed/60:.1f} min. best by val: {results[0]['tag']}")
    for r in results:
        print(f"  {r['tag']:24s} val={r['best_val_mean_geo']:.4f}  test={r['test_mean_geo']:.4f}")


if __name__ == '__main__':
    main()