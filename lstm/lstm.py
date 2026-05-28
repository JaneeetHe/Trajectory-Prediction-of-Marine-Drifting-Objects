"""LSTM for 24h drifter displacement prediction."""
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from metrics import evaluate


DATA_ROOT = Path(__file__).resolve().parent.parent / 'data_extended' # data directory
YEARS = [2022, 2023, 2024, 2025]

HISTORY_LEN = 8 # input length (8 steps = 48h)
HORIZON = 4 # output length (4 steps = 24h)
DT_SEC = HORIZON * 6.0 * 3600.0 # prediction horizon in seconds
MS_TO_DEG_LAT = DT_SEC / 111_320.0 # m/s to degrees latitude


# == Data Loading ==
def load_data(root=DATA_ROOT):
    Xs, ys = {s: [] for s in ('train', 'val', 'test')}, {s: [] for s in ('train', 'val', 'test')}
    feature_cols = None
    for year in YEARS:
        f = Path(root) / f'{year}_extended' / f'drifter_{year}_extended_supervised_windows.npz'
        d = np.load(f, allow_pickle=True)
        if feature_cols is None:
            feature_cols = [str(c) for c in d['feature_cols']]
        for s in ('train', 'val', 'test'):
            Xs[s].append(d[f'X_{s}_unscaled'])
            ys[s].append(d[f'y_{s}'])
    fi = {n: i for i, n in enumerate(feature_cols)} # feature index mapping
    out = {'feature_cols': feature_cols, 'fi': fi}
    for s in ('train', 'val', 'test'):
        X = np.concatenate(Xs[s], 0).astype('float32')
        y = np.concatenate(ys[s], 0).astype('float32')
        out[f'X_{s}_u'] = X
        out[f'y_{s}'] = y
        out[f'cur_lat_{s}'] = X[:, -1, fi['latitude']].astype('float32')
        out[f'cur_lon_{s}'] = X[:, -1, fi['longitude']].astype('float32')
    return out


# pack scaled X, raw y, advection prior, and current lat/lon for each split
def prepare_splits(data):
    fi = data['fi']
    scaler = make_scaler(data['X_train_u'], data['feature_cols'])
    sc = lambda x: apply_scaler(scaler, x)
    return dict(
        Xtr=sc(data['X_train_u']), Xva=sc(data['X_val_u']), Xte=sc(data['X_test_u']),
        ytr=data['y_train'], yva=data['y_val'], yte=data['y_test'],
        adv_tr=advection_delta(data['X_train_u'], fi),
        adv_va=advection_delta(data['X_val_u'], fi),
        adv_te=advection_delta(data['X_test_u'], fi),
        cur_lat_tr=data['cur_lat_train'], cur_lon_tr=data['cur_lon_train'],
        cur_lat_va=data['cur_lat_val'],   cur_lon_va=data['cur_lon_val'],
        cur_lat_te=data['cur_lat_test'],  cur_lon_te=data['cur_lon_test'],
    )


# == Data Scaling ==
def make_scaler(X_train_u, feature_cols):
    return StandardScaler().fit(X_train_u.reshape(-1, len(feature_cols)))


def apply_scaler(scaler, X):
    F_ = scaler.mean_.shape[0]
    return scaler.transform(X.reshape(-1, F_)).reshape(X.shape).astype('float32')


# == Physics Prior ==
# advection prediction: Δlat, Δlon from current ve/vn assuming constant velocity over 24h
def advection_delta(X_u, fi):
    last_lat = X_u[:, -1, fi['latitude']]
    ve = X_u[:, -1, fi['ve']]
    vn = X_u[:, -1, fi['vn']]
    cos_lat = np.cos(np.radians(last_lat))
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    dlat = vn * MS_TO_DEG_LAT
    dlon = ve * MS_TO_DEG_LAT / cos_lat
    return np.column_stack([dlat, dlon]).astype('float32')


# physics constraint penalty: hinge loss on predicted speeds above max_speed m/s
def physics_speed_penalty(pred_delta, cur_lat, max_speed=2.0):
    m_lon = (111_320.0 * torch.cos(torch.deg2rad(cur_lat))).clamp(min=1e-6)
    north = pred_delta[:, 0] * 111_320.0
    east  = pred_delta[:, 1] * m_lon
    speed = torch.sqrt(north**2 + east**2) / DT_SEC
    return torch.relu(speed - max_speed).mean()


# == LSTM Model ==
class DrifterLSTM(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, X, adv, physics_informed, device, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device)
            out.append(model(xb).cpu().numpy())
    net = np.concatenate(out, 0)
    return (net + adv).astype('float32') if physics_informed else net.astype('float32')


# == Training and Evaluation ==
DEFAULT_CFG = dict(
    input_dim=21, hidden_dim=128, num_layers=2, dropout=0.1,
    lr=1e-3, eta_min=1e-5, batch_size=256, epochs=50, patience=7,
    physics_informed=False, lambda_phys=0.0, seed=42, gpu_resident=True,
)


def train_model(d, cfg, label='LSTM'):
    cfg = {**DEFAULT_CFG, **cfg}
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    physics, lam = cfg['physics_informed'], cfg['lambda_phys']

    model = DrifterLSTM(cfg['input_dim'], cfg['hidden_dim'],
                        cfg['num_layers'], cfg['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=cfg['eta_min'])

    Xtr     = torch.from_numpy(d['Xtr']).to(device)
    ytr     = torch.from_numpy(d['ytr']).to(device)
    adv_tr  = torch.from_numpy(d['adv_tr']).to(device)
    lat_tr  = torch.from_numpy(d['cur_lat_tr']).to(device)
    g = torch.Generator(device=device); g.manual_seed(int(cfg['seed']))

    bs = cfg['batch_size']
    N = Xtr.shape[0]
    best_geo, best_state, best_epoch, bad = float('inf'), None, -1, 0
    history = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        idx = torch.randperm(N, generator=g, device=device)
        running, n_seen = 0.0, 0
        for i in range(0, N, bs):
            sel = idx[i:i+bs]
            xb, yb, advb, latb = Xtr[sel], ytr[sel], adv_tr[sel], lat_tr[sel]
            opt.zero_grad()
            net_out = model(xb)
            pred = net_out + advb if physics else net_out
            loss = F.mse_loss(pred, yb)
            if lam > 0:
                loss = loss + lam * physics_speed_penalty(pred, latb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n_seen += len(xb)
        sched.step()
        train_loss = running / n_seen

        y_va = predict(model, d['Xva'], d['adv_va'], physics, device)
        res, _ = evaluate(d['yva'], y_va, d['cur_lat_va'], d['cur_lon_va'], label)
        val_geo = float(res['mean_geo'])
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_mean_geo': val_geo})
        print(f"[{label}] ep {epoch:02d}  train {train_loss:.5f}  val {val_geo:.4f} km")

        if val_geo < best_geo - 1e-6:
            best_geo, best_epoch, bad = val_geo, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                print(f"[{label}] early stop @ ep {epoch}, best {best_geo:.4f} @ ep {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del Xtr, ytr, adv_tr, lat_tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, {'label': label, 'cfg': cfg, 'history': history,
                   'best_epoch': best_epoch, 'best_val_mean_geo': best_geo}
