# Port Intelligence System

An end-to-end machine learning system that predicts maritime port congestion using AIS vessel tracking and weather data. The project processes large-scale vessel movement data, engineers operational features, trains predictive models, and provides interactive insights through a deployed Streamlit dashboard.

## Live Demo

🔗 https://port-intelligence-system.streamlit.app

## Features

- Predicts port congestion risk using XGBoost
- Processes 690M+ AIS vessel records with memory-efficient PyArrow pipelines
- Engineers 34 operational and weather-based features
- SHAP-based prediction explainability
- 14-day congestion forecasting using Prophet
- Interactive Streamlit dashboard with geospatial visualizations and analytics

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy, PyArrow
- **Machine Learning:** XGBoost, Scikit-learn
- **Explainability:** SHAP
- **Forecasting:** Prophet
- **Visualization:** Plotly
- **Dashboard:** Streamlit

## Project Structure

```
port-intelligence/
├── dashboard/
├── src/
├── data/
├── models/
├── notebooks/
├── outputs/
└── requirements.txt
```

## Results

| Metric | Score |
|---------|------:|
| Mean LOPO AUC | 0.895 |

## Run Locally

```bash
git clone https://github.com/dhairyati/port-intelligence-system.git
cd port-intelligence-system

pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Future Improvements

- Real-time AIS data ingestion
- Support for additional global ports
- Docker deployment
- Automated model retraining