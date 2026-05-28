import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def geodesic_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def evaluate(y_true, y_pred, cur_lat, cur_lon, label='model'):
    mae_lat  = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_lon  = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_lat = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_lon = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    geo = geodesic_km(cur_lat + y_true[:, 0], cur_lon + y_true[:, 1],
                      cur_lat + y_pred[:, 0], cur_lon + y_pred[:, 1])
    return {
        'label': label,
        'mae_lat': mae_lat, 'mae_lon': mae_lon,
        'rmse_lat': rmse_lat, 'rmse_lon': rmse_lon,
        'mean_geo': geo.mean(),
        'median_geo': np.median(geo),
        'p90_geo': np.percentile(geo, 90),
    }, geo
