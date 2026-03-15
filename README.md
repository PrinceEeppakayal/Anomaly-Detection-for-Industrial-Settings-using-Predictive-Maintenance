# 5G Base Station Anomaly Detection (Supervised Learning)

This project simulates a telecom 5G base station and performs real-time anomaly detection using supervised machine learning.

It includes:

- A labeled data generator for 7 classes (normal + 6 anomaly types)
- A Random Forest training pipeline with epoch-style evaluation
- A real-time simulator that predicts anomaly classes live
- A Streamlit dashboard for monitoring metrics, confidence, and confusion matrix
- Optional email notifications for anomaly alerts

## Overview

The system monitors four core sensor signals:

- `temperature_C`
- `power_consumption_W`
- `signal_strength_dBm`
- `network_load_percent`

Anomaly classes used in the supervised workflow:

- `normal`
- `overheating`
- `power_surge`
- `signal_degradation`
- `network_overload`
- `cooling_failure`
- `equipment_failure`

## Project Structure

```text
anomaly-detection-using-predictive-maintenance
├── config/
│   └── email_config.json
├── data/
│   └── 5g_training_data_labeled.csv
├── docs/
│   └── ML_Model_Explanation.md
├── logs/
├── models/
├── notebooks/
│   └── poc_anomaly_dht11.ipynb
├── scripts/
│   ├── compare_models.py
│   ├── email_notifier.py
│   ├── generate_training_data.py
│   ├── realtime_5g_simulator_supervised.py
│   ├── run_supervised_dashboard.py
│   ├── streamlit_dashboard_supervised.py
│   └── train_model_with_epochs.py
├── dashboard_requirements.txt
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

For full training + simulator + dashboard:

```powershell
pip install -r requirements.txt
```

Dashboard-only setup:

```powershell
pip install -r dashboard_requirements.txt
```

### 3. Generate labeled training data

```powershell
python scripts/generate_training_data.py
```

This creates `data/5g_training_data_labeled.csv`.

### 4. Train the supervised model

```powershell
python scripts/train_model_with_epochs.py
```

This saves the trained model to `models/5g_supervised_model.pkl`.

### 5. Run the Streamlit dashboard

```powershell
python scripts/run_supervised_dashboard.py
```

Default dashboard URL: `http://localhost:8502`

### 6. (Optional) Run the simulator directly

```powershell
python scripts/realtime_5g_simulator_supervised.py
```

## Typical Workflow

```text
Generate dataset -> Train model -> Launch dashboard -> Monitor live predictions
```

If a trained model exists, the dashboard/simulator loads it automatically.
If not, it can fall back to training a new model from synthetic simulator data.

## Key Scripts

- `scripts/generate_training_data.py`
  - Creates balanced labeled samples and writes `data/5g_training_data_labeled.csv`
- `scripts/train_model_with_epochs.py`
  - Splits train/validation/test
  - Trains Random Forest over multiple epochs (increasing trees)
  - Saves best model and training artifacts
- `scripts/realtime_5g_simulator_supervised.py`
  - Simulates live telemetry and predicts anomaly class in real time
- `scripts/streamlit_dashboard_supervised.py`
  - Real-time dashboard with metrics, class predictions, confidence, and confusion matrix
- `scripts/run_supervised_dashboard.py`
  - Launcher that checks dependencies and starts Streamlit on port 8502
- `scripts/compare_models.py`
  - Compares supervised Random Forest against an unsupervised baseline
- `scripts/email_notifier.py`
  - Optional SMTP email alert system

## Email Alerts (Optional)

Configure SMTP credentials in `config/email_config.json`.

Important:

- Keep credentials private.
- For Gmail, use an app password (not your main account password).
- Set `enable_notifications` to `false` if you want to disable alerts.

## Outputs and Artifacts

- Dataset: `data/5g_training_data_labeled.csv`
- Trained model: `models/5g_supervised_model.pkl`
- Training plots (if generated):
  - `models/training_history.png`
  - `models/confusion_matrix.png`
- Runtime logs/prints: terminal output and optional log files in `logs/`

## Troubleshooting

### `ModuleNotFoundError` while running scripts

Reinstall dependencies in the active environment:

```powershell
pip install -r requirements.txt
```

### Dashboard does not open

- Confirm Streamlit is installed
- Ensure port `8502` is available
- Run the launcher from project root:

```powershell
python scripts/run_supervised_dashboard.py
```

### Model file not found

Run training first:

```powershell
python scripts/generate_training_data.py
python scripts/train_model_with_epochs.py
```

## Notes

- `docs/ML_Model_Explanation.md` contains a deeper explanation of model design and evaluation concepts.
- The notebook in `notebooks/poc_anomaly_dht11.ipynb` is available for experimentation, while the supervised scripts are the primary runnable pipeline in this workspace.

## Author

- Prince Eppakayal
- Khushi Khandari
- Akash Shukla

## License

Educational project for telecom/industrial monitoring and anomaly detection.
