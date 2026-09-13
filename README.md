# Customer Churn Prediction using Machine Learning

##  Project Overview

Customer churn is a major challenge for e-commerce businesses. Identifying customers who are likely to leave helps businesses improve customer retention, personalize offers, and develop effective engagement strategies.

This project focuses on building a machine learning model to predict whether an e-commerce customer will churn based on customer behavior, purchase history, engagement, and other relevant features.

The project uses **XGBoost Classification** to analyze customer data and identify customers who are at risk of churning.

##  Objectives

* Analyze customer behavior and identify churn patterns.
* Perform data preprocessing and exploratory data analysis.
* Build a machine learning model for customer churn prediction.
* Evaluate model performance using accuracy, precision, recall, and F1-score.
* Optimize the classification threshold to improve churn detection.
* Apply regularization and hyperparameter tuning to improve model performance.
* Understand how machine learning can support customer retention strategies.

##  Dataset

The project uses an e-commerce customer churn dataset.

| Property          | Details                           |
| ----------------- | --------------------------------- |
| Dataset           | E-commerce Customer Churn Dataset |
| Number of records | 50,000                            |
| Number of columns | 25                                |
| Problem type      | Binary Classification             |
| Target variable   | `Churned`                         |
| Target values     | 0 = Not Churned, 1 = Churned      |

### Dataset Features

The dataset contains customer-related information such as:

* Customer demographics
* Country
* Membership years
* Login frequency
* Purchase history
* Customer engagement
* Lifetime value
* Churn status

The target variable is `Churned`, which indicates whether a customer has left the platform.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Google Colab

## 🔍 Project Workflow

### 1. Data Loading

Loaded the e-commerce customer churn dataset using Pandas.

### 2. Data Preprocessing

* Loaded and inspected the dataset.
* Checked data types and missing values.
* Analyzed numerical and categorical features.
* Prepared the data for machine learning.

### 3. Exploratory Data Analysis

Performed exploratory analysis to understand customer churn patterns.

Visualizations included:

* Churn distribution.
* Customer lifetime value vs churn.
* Country-wise churn analysis.
* Correlation analysis.
* Outlier analysis.

### 4. Feature Preparation

Separated the input features and target variable.

```python
X = df.drop('Churned', axis=1)
y = df['Churned']
```

### 5. Model Building

Implemented an XGBoost classifier for customer churn prediction.

XGBoost was selected because it is a powerful gradient boosting algorithm suitable for structured/tabular datasets.

### 6. Model Evaluation

Evaluated the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 7. Threshold Optimization

Experimented with a classification threshold of `0.3` instead of the default `0.5`.

This was done to improve the detection of customers likely to churn.

### 8. Regularization

Implemented XGBoost regularization using:

* L1 Regularization (`reg_alpha`)
* L2 Regularization (`reg_lambda`)

### 9. Hyperparameter Tuning

Used `GridSearchCV` to optimize the XGBoost model parameters.

Parameters explored:

* `max_depth`
* `learning_rate`
* `n_estimators`

##  Model Results

### XGBoost Classification — Default Threshold

| Metric         |  Score |
| -------------- | -----: |
| Test Accuracy  | 91.49% |
| Train Accuracy | 96.28% |

### Classification Report

| Class           | Precision | Recall | F1-score |
| --------------- | --------: | -----: | -------: |
| 0 — Not Churned |      0.95 |   0.89 |     0.92 |
| 1 — Churned     |      0.77 |   0.87 |     0.82 |

**Churn Recall: 87%**

The model identified a substantial proportion of customers who churned, which is important for customer retention applications.

### Threshold-Tuned Model

| Metric                   |  Score |
| ------------------------ | -----: |
| Classification threshold |    0.3 |
| Test Accuracy            | 88.62% |
| Train Accuracy           | 94.58% |

The lower threshold prioritizes identifying more potential churners, with a trade-off in overall accuracy.

### Regularized XGBoost

| Metric        |               Score |
| ------------- | ------------------: |
| Test Accuracy |              92.20% |
| Model         | Regularized XGBoost |

### Hyperparameter Tuning

Best parameters reported by the notebook:

```python
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 200
}
```

## 💡 Key Learnings

* Customer churn prediction is a binary classification problem.
* Exploratory data analysis helps identify important customer behavior patterns.
* XGBoost performs well on structured customer datasets.
* Accuracy alone is not sufficient for evaluating churn prediction.
* Recall is important when the goal is to identify customers at risk of leaving.
* Classification threshold tuning can change the balance between precision and recall.
* Regularization and hyperparameter tuning can improve model performance.

## 🚀 Business Applications

This model can help e-commerce businesses:

* Identify customers at risk of churning.
* Develop targeted retention campaigns.
* Offer personalized discounts and recommendations.
* Improve customer engagement.
* Support data-driven customer retention strategies.

## 📁 Project Structure

```text
Customer-Churn-prediction/
│
├── README.md
├── customer_churn_prediction.ipynb
└── ecommerce_customer_churn_dataset.csv
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/bedkemanojkumar/Customer-Churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd Customer-Churn-prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Open the notebook

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Alternatively, open the notebook using Google Colab.

## 📌 Future Improvements

* Implement advanced feature engineering.
* Compare XGBoost with LightGBM and Random Forest.
* Perform cross-validation and systematic model evaluation.
* Improve recall and precision using threshold optimization.
* Deploy the model using Streamlit or Flask.
* Build a customer churn prediction dashboard.
* Add explainable AI using SHAP.
* Implement real-time customer churn risk prediction.

## 👨‍💻 Author

**Manojkumar Shriniwas Bedke**

B.Tech Computer Science Engineering — 2027

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Research.

### 🔗 Connect with me

* GitHub: https://github.com/bedkemanojkumar
* LinkedIn: https://linkedin.com/in/manojkumar-bedke01

---

⭐ If you find this project useful, consider giving the repository a star!
