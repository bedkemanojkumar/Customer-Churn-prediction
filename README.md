# Customer Churn Prediction using XGBoost

**An end-to-end machine learning project that predicts customer churn in an e-commerce environment using XGBoost, exploratory data analysis, threshold optimisation, regularization and hyperparameter tuning.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Pandas](https://img.shields.io/badge/Data-Pandas-purple)
![Jupyter](https://img.shields.io/badge/Environment-Jupyter-orange?logo=jupyter)

## Project Overview

Customer churn is a major challenge for e-commerce businesses. Identifying customers who are likely to leave a platform helps organisations improve customer retention, personalise engagement strategies and make data-driven business decisions.

This project develops a machine learning classification system to predict whether an e-commerce customer is likely to churn based on customer demographics, purchase history, engagement patterns and other behavioural attributes.

An **XGBoost classifier** is used to learn patterns associated with customer churn. The project explores the complete machine learning workflow, including data preprocessing, exploratory data analysis, model development, evaluation, classification threshold optimisation, regularization and hyperparameter tuning.

A key focus is understanding the trade-off between overall prediction accuracy and churn detection, where recall is particularly important for identifying customers at risk of leaving.

## Project Objectives

- Analyse customer behaviour and identify churn patterns.
- Perform data preprocessing and exploratory data analysis.
- Develop a supervised machine learning model for churn prediction.
- Evaluate model performance using classification metrics.
- Optimise the classification threshold to improve churn detection.
- Apply L1 and L2 regularization to control model complexity.
- Perform hyperparameter tuning using GridSearchCV.
- Explore how predictive modelling can support customer retention strategies.

## Dataset

The project uses an e-commerce customer churn dataset containing customer-related demographic, transactional and engagement information.

| Dataset Property | Details |
|---|---|
| Dataset | E-commerce Customer Churn Dataset |
| Number of Records | 50,000 |
| Number of Columns | 25 |
| Problem Type | Binary Classification |
| Target Variable | `Churned` |

### Features

The dataset includes customer attributes such as:

- Customer demographics.
- Country.
- Membership duration.
- Login frequency.
- Purchase history.
- Customer engagement.
- Customer lifetime value.
- Churn status.

### Target Variable

The target variable is `Churned`:

- `0` – Customer did not churn.
- `1` – Customer churned.

The task is formulated as a binary classification problem.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Classification Algorithm | XGBoost |
| Model Optimisation | GridSearchCV |
| Development Environment | Jupyter Notebook, Google Colab |

## Machine Learning Workflow

### 1. Data Loading and Inspection

- Loaded the customer churn dataset using Pandas.
- Inspected dataset dimensions, column names and data types.
- Examined missing values and feature distributions.
- Identified numerical and categorical attributes.

### 2. Data Preprocessing

Prepared the dataset for machine learning through data inspection, missing-value checks and feature preparation.

Separated the input features and target variable:

```python
X = df.drop('Churned', axis=1)
y = df['Churned']
```

### 3. Exploratory Data Analysis

Performed exploratory data analysis to understand customer behaviour and identify patterns associated with churn.

Visualisations included:

- Churn distribution.
- Customer lifetime value versus churn.
- Country-wise churn analysis.
- Feature correlation analysis.
- Outlier analysis.

EDA helped examine the relationships between customer characteristics and churn behaviour.

### 4. Model Development – XGBoost

Implemented an **XGBoost Classifier** for customer churn prediction.

XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees sequentially, with each boosting stage improving upon previous predictions.

It was selected for its effectiveness in modelling structured and tabular datasets.

### 5. Model Evaluation

Evaluated the classifier using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

Particular attention was given to recall for the churn class because failing to identify customers who are likely to leave can affect customer retention strategies.

### 6. Classification Threshold Optimisation

The default binary classification threshold is 0.5.

This project experiments with a lower threshold of **0.3** to investigate its effect on churn detection.

Reducing the threshold makes the classifier more likely to label a customer as a potential churner.

This can improve the identification of churn cases, but it may also increase false positives and reduce overall accuracy.

### 7. Regularization

Applied XGBoost regularization techniques to control model complexity:

| Parameter | Regularization |
|---|---|
| `reg_alpha` | L1 Regularization |
| `reg_lambda` | L2 Regularization |

These parameters help penalise model complexity and can reduce overfitting.

### 8. Hyperparameter Tuning

Used Scikit-learn's `GridSearchCV` to explore XGBoost hyperparameters and identify a better-performing configuration.

Parameters explored:

| Parameter | Description |
|---|---|
| `learning_rate` | Controls the contribution of each boosting iteration |
| `max_depth` | Controls the maximum depth of decision trees |
| `n_estimators` | Number of boosting estimators |

**Best parameters reported by the notebook:**

```python
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 200
}
```

## Model Performance and Results

The project evaluates the default XGBoost model, a threshold-adjusted model and a regularized XGBoost model.

### 1. Default XGBoost Model

<details>
<summary>Performance metrics</summary>

| Metric | Score |
|---|---:|
| Training Accuracy | 96.28% |
| Test Accuracy | 91.49% |
| Churn Recall | 87% |

**Classification Report**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 – Not Churned | 0.95 | 0.89 | 0.92 |
| 1 – Churned | 0.77 | 0.87 | 0.82 |

The model achieved 91.49% test accuracy and identified approximately 87% of actual churn cases.

</details>

### 2. Threshold-Optimised Model

<details>
<summary>Performance metrics</summary>

| Metric | Score |
|---|---:|
| Classification Threshold | 0.30 |
| Training Accuracy | 94.58% |
| Test Accuracy | 88.62% |

The lower threshold prioritises identifying more potential churners, with a trade-off in overall accuracy.

</details>

### 3. Regularized XGBoost Model

<details>
<summary>Performance metrics</summary>

| Metric | Score |
|---|---:|
| Test Accuracy | 92.20% |
| Model | Regularized XGBoost |

The regularized model achieved the highest reported test accuracy among the documented configurations.

</details>

### Model Comparison

| Model Configuration | Test Accuracy | Key Observation |
|---|---:|---|
| Default XGBoost | 91.49% | Churn recall of 87% |
| Threshold 0.3 | 88.62% | Prioritises churn detection |
| Regularized XGBoost | **92.20%** | Highest reported test accuracy |

**Key observation:** The project demonstrates that model evaluation should consider the business objective rather than relying exclusively on accuracy. Threshold selection changes the balance between detecting churners and incorrectly flagging customers who would remain.

*Note: The results above are reported from the project notebook. The threshold-adjusted model's churn recall and the regularized model's class-specific metrics should be reported separately if available.*

## Business Applications

The model can support e-commerce customer retention strategies by helping businesses:

- Identify customers with a higher likelihood of churning.
- Prioritise customer retention campaigns.
- Develop personalised discounts and offers.
- Improve customer engagement.
- Analyse customer behaviour and retention patterns.
- Support data-driven marketing decisions.

The predictions can serve as inputs to customer retention workflows, where business teams decide which interventions are appropriate.

## Project Structure

```text
Customer-Churn-prediction/
│
├── README.md
│
├── customer_churn_prediction.ipynb
│
└── ecommerce_customer_churn_dataset.csv
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bedkemanojkumar/Customer-Churn-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Customer-Churn-prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Alternatively, upload the notebook to Google Colab and execute its cells.

## Key Learnings

Through this project, I gained practical experience in:

- Solving binary classification problems using machine learning.
- Performing exploratory data analysis on a large customer dataset.
- Building gradient boosting models using XGBoost.
- Evaluating classification performance beyond accuracy.
- Understanding precision–recall trade-offs.
- Optimising classification thresholds for business objectives.
- Applying L1 and L2 regularization.
- Performing hyperparameter tuning using GridSearchCV.
- Translating machine learning predictions into potential business applications.

## Future Improvements

- Implement advanced feature engineering.
- Compare XGBoost with LightGBM and Random Forest.
- Perform systematic cross-validation and model evaluation.
- Optimise precision and recall based on retention objectives.
- Add SHAP-based feature importance and prediction explanations.
- Develop an interactive Streamlit or Flask dashboard.
- Implement real-time customer churn risk prediction.
- Integrate model monitoring and experiment tracking.

## Author

**Manojkumar Shriniwas Bedke**

B.Tech – Computer Science Engineering, 2027

Interested in Artificial Intelligence, Machine Learning, Deep Learning and Applied AI Research.

- **GitHub:** [bedkemanojkumar](https://github.com/bedkemanojkumar)
- **LinkedIn:** [manojkumar-bedke01](https://linkedin.com/in/manojkumar-bedke01)

---

*This project demonstrates the application of machine learning and model optimisation techniques to customer retention and churn prediction problems.*
