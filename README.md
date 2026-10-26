# NYC-Taxi-Demand-Forecasting-Operations-Intelligence-Platform

**Technologies:** Python, SQL, DuckDB, Pandas, Scikit-learn, LightGBM, FastAPI, Docker, Streamlit, MLflow

* Engineered an end-to-end data processing pipeline using **Python, SQL, and DuckDB** to ingest, clean, and transform large-scale NYC taxi trip records, implementing data quality checks and aggregating historical pickup activity into location-based, hourly datasets for predictive analytics.

* Developed and optimized **machine learning forecasting models using LightGBM, Random Forest, and Scikit-learn**, incorporating historical demand patterns, lag features, rolling averages, and temporal variables to predict hourly taxi pickup volumes across geographic zones.

* Implemented **time-series cross-validation and rolling backtesting** to evaluate model generalization while preventing future-data leakage; benchmarked forecasting performance against historical baselines using **MAE, RMSE, and WAPE**, achieving a **[X%] reduction in forecasting error** over the selected baseline.

* Designed an interactive **operations intelligence dashboard using Streamlit and FastAPI**, integrating model predictions with geographic demand visualizations to identify anticipated pickup-volume increases and support data-driven vehicle allocation and operational planning.

* Established a reproducible **MLOps workflow using MLflow and Docker**, incorporating experiment tracking, model versioning, containerized inference, and automated data validation to support consistent model evaluation and deployment.
