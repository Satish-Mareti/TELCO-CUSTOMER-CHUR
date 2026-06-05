# Telco Customer Churn Pipeline End-to-end machine learning pipeline for the Kaggle / IBM Telco Customer Churn dataset. ## What this project does - Runs EDA on the cleaned dataset and saves a written summary. - Engineers features from the raw Telco columns. - Trains and compares three models with stratified cross-validation. - Handles imbalance with SMOTE and class-weighted estimators. - Tunes the strongest model with randomized search. - Saves the final model bundle with joblib. - Exposes a simple CLI for churn predictions. ## Expected dataset Download the Telco Customer Churn CSV from Kaggle and place it anywhere convenient, then pass the path to train.py. The CSV should contain the standard Telco columns, including: - customerID - gender - SeniorCitizen - Partner - Dependents - tenure - PhoneService - MultipleLines - InternetService - OnlineSecurity - OnlineBackup - DeviceProtection - TechSupport - StreamingTV - StreamingMovies - Contract - PaperlessBilling - PaymentMethod - MonthlyCharges - TotalCharges - Churn ## Install
bash
pip install -r requirements.txt
If your environment uses the provided typo file, this also works:
bash
pip install -r requriments.txt
## Train
bash
python train.py --data-path path/to/Telco-Customer-Churn.csv
Artifacts are written to artifacts/: - telco_churn_model.joblib - eda_summary.md - model_comparison.csv ## Predict Interactive mode:
bash
python app.py
JSON mode:
bash
python app.py --input-json '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.0,"TotalCharges":70.0}'
## Notes - The model is trained on the standard Telco churn schema; if your CSV uses different column names, align them first. - SMOTE is applied only on the training folds and training split, not on the final test set. - The saved bundle contains the trained pipeline, comparison table, and test metrics. # Author : Mareti Satish
