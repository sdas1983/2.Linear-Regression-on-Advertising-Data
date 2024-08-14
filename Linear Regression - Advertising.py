# Linear Regression on Advertising Data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ydata_profiling import ProfileReport
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, ElasticNet, ElasticNetCV

# Load the Advertising dataset
df = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Advertising.csv")

# Dataset Overview
print("\n### Dataset Information")
df.info()
print("\n### Dataset Description")
print(df.describe())
print("\n### Missing Values Count")
print(df.isnull().sum())
print("\n### Data Types")
print(df.dtypes)
print("\n### Dataset Shape")
print(df.shape)

# Drop unnecessary columns
df.drop(columns='ID', inplace=True)

# Data Visualization
print("\n### Boxplot for the Advertising Data")
sns.boxplot(data=df)
plt.title("Boxplot for the Advertising Data")
plt.show()

# Feature Selection
print("\n### Correlation Heatmap")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Advertising Features")
plt.show()

# Define features and target variable
X = df.drop(columns=['Sales'])
y = df['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save the model
pickle.dump(lr, open('Advertising_lr_model.pickle', 'wb'))

# Load and use the saved model
model = pickle.load(open('Advertising_lr_model.pickle', 'rb'))

# Predict on new data
print("\n### Predictions on New Data")
test_1 = scaler.transform([[230.1, 37.8, 69.2]])
test_2 = scaler.transform([[44.5, 39.3, 45.1]])
test_3 = scaler.transform([[17.2, 45.9, 69.3]])

print(f"Prediction for test_1: {model.predict(test_1)}")
print(f"Prediction for test_2: {model.predict(test_2)}")
print(f"Prediction for test_3: {model.predict(test_3)}")

# Evaluate the model
print("\n### Model Evaluation")
print(f"Test Score: {lr.score(X_test, y_test)}")
print(f"Train Score: {lr.score(X_train, y_train)}")

# Coefficients and Intercept
print("\n### Model Coefficients and Intercept")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")

# Cross-validation for Mean Squared Error (MSE)
print("\n### Cross-validation for Mean Squared Error (MSE)")
mse = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
print(f"Mean Squared Error: {np.mean(mse)}")

# Load Admission Predict dataset
df_admission = pd.read_csv("https://raw.githubusercontent.com/divyansha1115/Graduate-Admission-Prediction/master/Admission_Predict.csv")

# Dataset Overview
print("\n### Admission Prediction Dataset Information")
df_admission.info()
print("\n### Admission Prediction Dataset Description")
print(df_admission.describe())
print("\n### Missing Values Count")
print(df_admission.isnull().sum())
print("\n### Data Types")
print(df_admission.dtypes)
print("\n### Dataset Shape")
print(df_admission.shape)

# Drop unnecessary columns
df_admission.drop(columns='Serial No.', inplace=True)

# Data Visualization
print("\n### Boxplots for Admission Predict Data")
sns.boxplot(data=df_admission[['GRE Score']])
plt.title("Boxplot for GRE Score")
plt.show()

sns.boxplot(data=df_admission[['TOEFL Score']])
plt.title("Boxplot for TOEFL Score")
plt.show()

sns.boxplot(data=df_admission)
plt.title("Boxplot for All Features")
plt.show()

# Feature Selection
print("\n### Correlation Heatmap for Admission Predict Data")
sns.heatmap(df_admission.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Admission Predict Features")
plt.show()

# Define features and target variable
X = df_admission.drop(columns=['Chance of Admit '])
y = df_admission['Chance of Admit ']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Feature Scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression model
lr.fit(X_train, y_train)

# Save the model
pickle.dump(lr, open('Admission_Predict_lr_model.pickle', 'wb'))

# Load and use the saved model
model = pickle.load(open('Admission_Predict_lr_model.pickle', 'rb'))

# Predict on new data
print("\n### Predictions on Admission Data")
test_1 = scaler.transform(X.loc[0].values.reshape(1, -1))
test_2 = scaler.transform(X.loc[1].values.reshape(1, -1))
test_3 = scaler.transform(X.loc[2].values.reshape(1, -1))

print(f"Prediction for test_1: {model.predict(test_1)}")
print(f"Prediction for test_2: {model.predict(test_2)}")
print(f"Prediction for test_3: {model.predict(test_3)}")

# Evaluate the model
print("\n### Model Evaluation for Admission Predict")
print(f"Test Score: {lr.score(X_test, y_test)}")
print(f"Train Score: {lr.score(X_train, y_train)}")

# Coefficients and Intercept
print("\n### Model Coefficients and Intercept for Admission Predict")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")

# Cross-validation for Mean Squared Error (MSE)
print("\n### Cross-validation for Mean Squared Error (MSE) for Admission Predict")
mse = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
print(f"Mean Squared Error: {np.mean(mse)}")

# Cross-validation for Root Mean Squared Error (RMSE)
print("\n### Cross-validation for Root Mean Squared Error (RMSE)")
rmse = cross_val_score(lr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10)
print(f"Root Mean Squared Error: {np.mean(rmse)}")

# Cross-validation for Mean Absolute Error (MAE)
print("\n### Cross-validation for Mean Absolute Error (MAE)")
mae = cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print(f"Mean Absolute Error: {np.mean(mae)}")

# Generate a data profiling report
print("\n### Generating Data Profiling Report")
profile = ProfileReport(df_admission, title='Admission Prediction Data Profiling Report')
profile.to_file("admission_data_report.html")

# Model Selection using Variance Inflation Factor (VIF)
print("\n### Variance Inflation Factor (VIF) Analysis")
vif_df = pd.DataFrame()
vif_df['vif'] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
vif_df['feature'] = X.columns
print(vif_df)

# Calculate R-squared and Adjusted R-squared
r_squared = lr.score(X_train, y_train)

def adj_r2(x, y):
    r2 = lr.score(x, y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

print(f"Adjusted R-squared: {adj_r2(X_test, y_test)}")

# Lasso Regression
print("\n### Lasso Regression")
lassocv = LassoCV(cv=10, max_iter=200000)
lassocv.fit(X_train, y_train)
lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)

print(f"Lasso Train Score: {lasso.score(X_train, y_train)}")
print(f"Lasso Test Score: {lasso.score(X_test, y_test)}")

# Ridge Regression
print("\n### Ridge Regression")
ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10)
ridgecv.fit(X_train, y_train)
ridge_lr = Ridge(alpha=ridgecv.alpha_)
ridge_lr.fit(X_train, y_train)

print(f"Ridge Train Score: {ridge_lr.score(X_train, y_train)}")
print(f"Ridge Test Score: {ridge_lr.score(X_test, y_test)}")

# Elastic Net Regression
print("\n### Elastic Net Regression")
elastic = ElasticNetCV(alphas=None, cv=10)
elastic.fit(X_train, y_train)
elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio_)
elastic_lr.fit(X_train, y_train)

print(f"Elastic Net Train Score: {elastic_lr.score(X_train, y_train)}")
print(f"Elastic Net Test Score: {elastic_lr.score(X_test, y_test)}")

# Cross-validation for Elastic Net MSE
print("\n### Cross-validation for Elastic Net MSE")
elastic_lr_mse = cross_val_score(elastic_lr, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
print(f"Elastic Net Mean Squared Error: {np.mean(elastic_lr_mse)}")

# Cross-validation for Elastic Net RMSE
print("\n### Cross-validation for Elastic Net RMSE")
elastic_lr_rmse = cross_val_score(elastic_lr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10)
print(f"Elastic Net Root Mean Squared Error: {np.mean(elastic_lr_rmse)}")

# Cross-validation for Elastic Net MAE
print("\n### Cross-validation for Elastic Net MAE")
elastic_lr_mae = cross_val_score(elastic_lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
print(f"Elastic Net Mean Absolute Error: {np.mean(elastic_lr_mae)}")
