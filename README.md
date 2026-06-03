# 🌊 Trajectory Prediction of Marine Drifting Objects

> **ECE/SIOC 228 Final Project — Team 32, UC San Diego (May 2026)**  
> Ruixin Fu · Yiting Wang · Jingyi He · Danni Ma

---

## 📌 Project Overview

This project addresses the problem of predicting the future trajectory of marine drifting objects (ocean buoys, floating debris, etc.) given a sequence of historical observations.

- **Data**: NOAA Global Drifter Program 6-hour interpolated buoy observations (2022–2025)
- **Input**: Past 8 time steps × 21 features (= 48 hours of history)
- **Output**: Future displacement (Δlat, Δlon) after 24 hours
- **Core Question**: Can ML systematically outperform physics-based baselines?

**Why it matters:** Accurate drift prediction supports search & rescue operations, environmental monitoring, oil spill response, and marine debris tracking.

---

## 📁 Repository Structure

```
├── data_processing.ipynb        # Raw NetCDF → supervised windows
├── baseline_models.ipynb        # Physics & classical ML baselines
├── lstm_experiments.ipynb       # LSTM variants (vanilla / phys-residual / phys-res+loss)
├── baseline_metrics.csv         # Evaluation results for baseline models
├── loss_1.png                   # Training loss curve (experiment 1)
├── loss_2.png                   # Training loss curve (experiment 2)
└── README.md
```

---

## 🗂️ Dataset

Raw data is organized by year under `raw_Data/` and processed into supervised windows under `processed_Data/`:

```
raw_Data/
    2022/  2023/  2024/  2025/       # .nc files per drifter trajectory

processed_Data/
    <year>_extended/
        drifter_<year>_extended_supervised_windows.npz   # Main training arrays
        drifter_<year>_extended_window_metadata.csv
        drifter_<year>_extended_processing_config.json
        drifter_<year>_extended_file_log.csv
```

Each `.npz` file contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `X_train` | (N, 8, 21) | Standardized training input |
| `y_train` | (N, 2) | Training targets (Δlat, Δlon) |
| `X_val` | (N, 8, 21) | Validation input |
| `y_val` | (N, 2) | Validation targets |
| `X_test` | (N, 8, 21) | Test input |
| `y_test` | (N, 2) | Test targets |

**Train / Val / Test split**: 70% / 15% / 15%

### Feature Groups (21 total)

| Group | Features |
|-------|----------|
| Core trajectory | `latitude`, `longitude`, `ve`, `vn` |
| Optional raw | `temp`, `err_lat`, `err_lon`, `err_temp` |
| Constructed physical | `speed`, `direction`, `delta_lat_step`, `delta_lon_step`, `delta_ve_step`, `delta_vn_step`, `accel_east`, `accel_north`, `step_distance_km` |
| Time encodings | `doy_sin`, `doy_cos`, `hour_sin`, `hour_cos` |

---

## 🧠 Methods

### Tier 1 — Physics Baselines
- **Persistence**: assumes the drifter stays at its current position
- **Advection**: `x_{t+Δ} = x_t + v_t · Δt`

### Tier 2 — Classical ML
- **Ridge Regression** and **Random Forest** on flattened (N, T·F) input

### Tier 3 — Physics-Constrained ML
- **PC-RF**: Random Forest trained on advection residuals (physics-informed residual learning)

### Deep Learning
- **LSTM (vanilla)**: 2-layer LSTM encoder (hidden dim=128) + MLP head (128→64→2)
- **LSTM (phys-residual)**: adds advection prior — `Δ_pred = Δ_res + Δ_adv`
- **LSTM (phys-res+loss)**: adds physics penalty term to the loss: `L = MSE + λ · L_phys`
- **Transformer (phys-residual)**: multi-head attention (nhead=4) + Neural ODE + MLP decoder

---

## 📊 Results

### Baseline Models

| Model | MAE lat (°) | MAE lon (°) | Mean Geo (km) | Median Geo (km) | P90 Geo (km) |
|-------|-------------|-------------|---------------|-----------------|--------------|
| Persistence (Tier 1) | 0.0980 | 0.1640 | 19.11 | 14.21 | 39.72 |
| Advection (Tier 1) | 0.0510 | 0.0727 | 8.72 | 6.61 | 17.25 |
| Ridge (Tier 2) | 0.0456 | 0.0651 | 7.80 | 5.99 | 15.25 |
| Random Forest (Tier 2) | 0.0428 | 0.0618 | 7.25 | 5.49 | 14.43 |
| PC-RF (Tier 3) | 0.0402 | 0.0591 | 6.86 | 5.15 | 13.67 |

### Deep Learning Models

| Model | Mean Geo (km) | Median Geo (km) |
|-------|---------------|-----------------|
| LSTM (vanilla) | 6.07 | 4.49 |
| LSTM (phys-residual) | **5.88** | **4.28** |
| LSTM (phys-res+loss) | 5.88 | 4.28 |
| Transformer (phys-residual) | 8.57 | 5.03 |

### Key Findings

1. **LSTM achieves best overall accuracy** — mean geodesic error of **5.88 km**
2. **Residual prior helps** (6.07 → 5.88 km); adding a physics loss term provides no further gain
3. **Transformer underperforms LSTM** on short sequences — recurrence is better suited here
4. **Longitude error > Latitude error** consistently across all models
5. **Physics + ML outperforms physics alone** at every tier

---

## ⚙️ Environment

```bash
pip install numpy pandas scikit-learn torch torchdiffeq matplotlib jupyter
```

Main dependencies: `Python 3.9+`, `PyTorch`, `scikit-learn`, `NumPy`, `pandas`

---

## 🚀 Quick Start

```python
import numpy as np

# Load processed data
data = np.load("processed_Data/2022_extended/drifter_2022_extended_supervised_windows.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

print(X_train.shape)  # (N, 8, 21)
print(y_train.shape)  # (N, 2)
```

Run the notebooks in order:
1. `data_processing.ipynb` — preprocess raw NetCDF files
2. `baseline_models.ipynb` — train and evaluate physics & ML baselines
3. `lstm_experiments.ipynb` — train LSTM/Transformer variants

---

## 📖 References

1. Jansen et al. (2016). Drift simulation of MH370 debris using superensemble techniques. *NHESS*.
2. Griffin (2017). The search for MH370 and ocean surface drift. *CSIRO Report*.
3. Delandmeter & van Sebille (2019). The Parcels v2.0 Lagrangian framework. *Geosci. Model Dev.*
4. Yan et al. (2023). Sea drift trajectory prediction based on quantum CNN-LSTM. *Applied Sciences*.
5. Botvynko et al. (2025). Neural prediction of Lagrangian drift trajectories. *AI for Earth Systems*.
6. Lumpkin & Centurioni (2019). Global Drifter Program quality-controlled 6-hour data. *NOAA NCEI*.

---

## 📜 License

This project is for academic use only as part of the ECE/SIOC 228 course at UC San Diego.
