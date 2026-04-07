# 💳 Customer Churn Prediction & Retention Optimization

## 📌 Business Problem

Customer churn is one of the biggest challenges in the banking industry.
Losing customers directly impacts revenue and increases acquisition costs.

**Goal:**
Build a machine learning system to identify customers at high risk of churn and optimize retention campaigns.

---

## 🎯 Project Objectives

* Predict customer attrition (churn vs. non-churn)
* Maximize **recall** to avoid missing high-risk customers
* Optimize **business profit**, not just model accuracy
* Simulate real-world decision making (A/B testing)

---

## 🧠 Approach

### 1. Data Preparation

* Missing value imputation
* Feature scaling for numerical variables
* One-hot encoding for categorical variables
* Pipeline-based preprocessing (production-ready)

---

### 2. Modeling

Models evaluated:

* Logistic Regression
* Random Forest (final model)

Key techniques:

* Stratified train-test split
* Cross-validation (5-fold)
* Hyperparameter tuning (RandomizedSearchCV)

---

### 3. Experiment Tracking

All experiments tracked using **MLflow**:

* Model parameters
* Metrics (ROC-AUC, CV scores)
* Model artifacts

---

### 4. Business-Oriented Optimization

Instead of using default predictions:

✔ Custom probability threshold
✔ Focus on **recall (churn detection)**

```python
threshold = 0.3
y_pred = (y_proba > threshold).astype(int)
```

---

### 5. A/B Testing Simulation (🔥 Key Insight)

We simulate two strategies:

| Strategy    | Description                |
| ----------- | -------------------------- |
| Baseline    | Random customer contact    |
| Model-Based | Target high-risk customers |

#### Business assumptions:

* Contact cost: $10
* Retention benefit: $200

#### Result:

👉 The model-driven strategy generates higher profit and reduces wasted contacts.

---

## 📊 Key Metrics

* ROC-AUC
* Recall (primary metric)
* Precision
* Cross-validation stability

---

## 💰 Business Impact

✔ Better targeting of retention campaigns
✔ Reduced operational costs
✔ Increased customer lifetime value

👉 The model improves decision-making by focusing on **who to contact and why**

---

## 🛠️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* MLflow

---

## 📁 Project Structure

```
churn_project/
│
├── data/
├── src/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

Run MLflow UI:

```bash
mlflow ui
```

---

## 🧠 Key Learnings

* Accuracy is not enough → business metrics matter
* Threshold tuning is critical in imbalanced problems
* Experiment tracking improves reproducibility
* ML must align with real-world decision making

---

## 🔥 Next Steps

* SHAP for model interpretability
* Deployment (API or batch scoring)
* Real A/B testing in production environment

---

## 👤 Author

Data-focused project combining machine learning and business strategy.
