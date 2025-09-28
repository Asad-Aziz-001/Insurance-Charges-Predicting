# Insurance-Charges-Predicting
This project predicts medical insurance charges using client attributes such as age, BMI, children, smoking status, and region. Regression models (Linear, Random Forest, XGBoost) were trained, with XGBoost achieving the best accuracy. Visualizations and a Streamlit app provide an interactive, modern UI for predictions.

[![Streamlit Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://insurance-charges-predicting-epc42axmtwzapp5wurzckky.streamlit.app/)


# **📑 Project Report: Predicting Insurance Charges Using Machine Learning**

# **1. Objective**

The main goal of this project is to predict medical insurance charges for clients based on their personal and health-related attributes such as:

Age

BMI (Body Mass Index)

Number of children

Smoking status

Gender and region (if included in the dataset)

This helps insurance companies estimate fair premiums and allows individuals to understand how their lifestyle and health choices impact costs.

# **2. Dataset**

We used the “Prediction of Insurance Charges” dataset from Kaggle.

Each row represents an individual insured person.

The target variable is charges (medical insurance cost).

Features include demographic details (age, sex, region), lifestyle factors (smoking), and physical health indicators (BMI, number of children).

**Key properties of the dataset:**

Size: ~1,300 records

Mix of numerical (age, BMI, children) and categorical (sex, smoker, region) variables

Target variable (charges) is right-skewed — most people have moderate charges, but smokers or high-BMI individuals may have very high charges.

# **3. Methodology**

*Step 1: Data Exploration (EDA)*

Checked missing values (none or handled appropriately).

Visualized distributions:

Charges: right-skewed distribution.

Smoker vs non-smoker: smokers had dramatically higher charges.

BMI: higher BMI correlated with higher charges, especially among smokers.

Age: charges increase steadily with age.

*Step 2: Data Preprocessing*

Numerical features:

Imputed missing values with median (if any).

Standardized (scaled) for models sensitive to feature scale.

Categorical features:

Converted into numeric format using one-hot encoding (e.g., smoker → yes/no).

Target variable:

Tried both raw charges and log-transformed charges (to reduce skewness).

*Step 3: Model Selection*

We trained three regression models:

Linear Regression (baseline model)

Simple, interpretable, but limited in capturing non-linear effects.

Random Forest Regressor

Ensemble of decision trees, good for non-linear relationships and feature interactions.

XGBoost Regressor

Gradient boosting model, highly effective for structured/tabular data, often outperforming others.

*Step 4: Model Training & Hyperparameter Tuning*

Used train-test split (80% train, 20% test).

Applied GridSearchCV with 5-fold cross-validation to tune hyperparameters.

Evaluation metrics:

MAE (Mean Absolute Error) → average error in cost prediction.

RMSE (Root Mean Squared Error) → penalizes larger errors.

R² (Coefficient of Determination) → how well the model explains variance in charges.

# **4. Results**

| Model             | MAE (↓) | RMSE (↓) | R² (↑) |
| ----------------- | ------- | -------- | ------ |
| Linear Regression | ~4187   | ~5798    | 0.78   |
| Random Forest     | ~1571   | ~4432    | 0.87   |
| XGBoost           | ~1836   | ~4237    | 0.88   |

Model Performance (with log-transform on target):

XGBoost (log-transformed) → MAE improved further to ~1726.

Interpretation:

Linear Regression underperformed (R² = 0.78) because insurance costs have non-linear patterns (e.g., smoking effect is not linear).

Random Forest and XGBoost captured these non-linearities much better.

Best overall model → XGBoost with log-transform, achieving the lowest MAE and highest R².

# **5. Predictions Example**

Sample predictions from the best model:

[ 9384,  5373, 28397,  8996, 34385 ]


These are predicted insurance charges for 5 unseen test records.

Moderate predictions (~5k–9k) → younger, healthier individuals.

Very high predictions (~28k–34k) → likely older smokers with high BMI.

# **6. Insights**

Smoking status is the strongest predictor — smokers consistently had much higher charges.

BMI significantly impacts costs, especially in combination with smoking.

Age is positively correlated with charges — older individuals incur higher costs.

Number of children and region have weaker impacts.

# **7. Visualization (Key Graphs)**

Distribution of charges → heavily right-skewed.

Charges by smoker status → smokers’ median charges are several times higher.

Actual vs Predicted charges → tree models closely track true values, unlike linear regression.

Feature importances (from RF/XGB) → top factors: smoker, age, BMI.

Residual plots → show reduced bias in XGBoost vs linear regression.

# **8. Model Deployment**

Saved the final tuned XGBoost pipeline (including preprocessing) using joblib.

Model can be loaded later and used to predict charges on new client data.

Possible deployment options:

FastAPI/Flask API → real-time predictions via JSON inputs.

Streamlit dashboard → interactive web app for visualization + prediction.

# **9. Conclusion**

XGBoost with log-transformed target was the most accurate model.

Achieved MAE ≈ 1726 and R² ≈ 0.88, meaning the model explains ~88% of the variance in insurance charges.

Smoking, BMI, and age are the dominant factors influencing costs.

The project demonstrates how machine learning can support fairer insurance pricing and provide clients with data-driven insights into their health-related risks.

# **10. Future Work**

Expand dataset with more health indicators (blood pressure, diabetes, exercise, etc.).

Use stacked ensemble models (combining RF + XGBoost + linear regression).

Predict confidence intervals (e.g., “expected charge: $9,000 ± $1,200”).

Deploy as a web app for public interaction.
