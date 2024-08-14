# 2.Linear Regression on Advertising and Admission Data

This project demonstrates the application of Linear Regression and other regression techniques on two datasets: Advertising and Graduate Admission Prediction. The project includes data preprocessing, feature selection, model training, evaluation, and model saving using Python.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Data Preprocessing](#data-preprocessing)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Advanced Regression Techniques](#advanced-regression-techniques)
- [Model Saving and Loading](#model-saving-and-loading)
- [Data Profiling](#data-profiling)
- [Variance Inflation Factor (VIF)](#variance-inflation-factor-vif)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This project performs linear regression on two datasets:

1. **Advertising Data**: Predicting sales based on advertising expenditures on TV, radio, and newspapers.
2. **Graduate Admission Prediction**: Predicting the likelihood of admission based on various factors such as GRE Score, TOEFL Score, University Rating, etc.

## Dataset Information

- **Advertising Dataset**: Contains the following features:
  - TV: Advertising budget spent on TV.
  - Radio: Advertising budget spent on Radio.
  - Newspaper: Advertising budget spent on Newspaper.
  - Sales: The target variable, representing sales.

- **Graduate Admission Dataset**: Contains the following features:
  - GRE Score: Graduate Record Examination score.
  - TOEFL Score: Test of English as a Foreign Language score.
  - University Rating: Rating of the university.
  - SOP: Statement of Purpose strength.
  - LOR: Letter of Recommendation strength.
  - CGPA: Undergraduate GPA.
  - Research: Whether the student has research experience (0 or 1).
  - Chance of Admit: The target variable, representing the likelihood of admission.

## Data Preprocessing

- **Handling Missing Values**: Checked for and handled any missing values in the datasets.
- **Data Types and Conversion**: Ensured correct data types and performed any necessary conversions.
- **Feature Selection**: Selected relevant features based on correlation analysis.

## Modeling and Evaluation

- **Linear Regression**: Trained a Linear Regression model on both datasets.
- **Model Evaluation**: Evaluated the models using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
- **Model Coefficients**: Retrieved and analyzed the coefficients and intercepts of the models.

## Advanced Regression Techniques

- **Lasso Regression**: Applied Lasso regression to perform feature selection and prevent overfitting.
- **Ridge Regression**: Applied Ridge regression for regularization.
- **ElasticNet Regression**: Combined the penalties of Lasso and Ridge regression.

## Model Saving and Loading

- **Model Persistence**: Saved the trained models using `pickle` for future use.
- **Model Loading**: Demonstrated how to load and use the saved models to make predictions.

## Data Profiling

- **Pandas Profiling**: Generated detailed data profiling reports using `pandas-profiling` to understand the dataset better.

## Variance Inflation Factor (VIF)

- **VIF Calculation**: Calculated the Variance Inflation Factor to check for multicollinearity among features.

## Installation

Ensure that you have Python installed. Install the required libraries using the following command:

```python
pip install -r requirements.txt
```

## Usage
Run the Python scripts to execute the linear regression analysis on the datasets. The code will output model evaluation metrics, plots, and predictions based on test data.
