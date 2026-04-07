# 💳 Churn Prediction & Revenue Optimization for Banking

## 🚨 Why This Project Matters

Customer churn is not just a modeling problem — it's a **direct revenue leakage problem**.

In retail banking (e.g., BBVA, Santander), poorly targeted retention campaigns lead to:

* ❌ Wasted marketing budget
* ❌ Missed high-value customers at risk
* ❌ Inefficient CRM strategies

👉 This project reframes churn prediction as a **profit optimization problem**, not a classification task.

---

## 🎯 Objective

Build an end-to-end ML system to:

* Identify high-risk churn customers
* Optimize retention campaign targeting
* Maximize **expected profit**, not accuracy

---

## 🧠 Key Differentiator

Most churn models optimize:

> ❌ Accuracy

This project optimizes:

> ✅ **Business impact (Profit + Recall)**

---

## ⚙️ Solution Overview

### 🔹 1. Production-Ready Pipeline

* Full preprocessing pipeline (scaling + encoding)
* No data leakage
* Reproducible workflow

---

### 🔹 2. Model Development

Models tested:

* Logistic Regression
* Random Forest (selected)

Techniques:

* Stratified split
* Cross-validation (5-fold)
* Hyperparameter tuning (RandomizedSearchCV)

---

### 🔹 3. Experiment Tracking

All experiments tracked with MLflow:

* Parameters
* ROC-AUC
* Cross-validation metrics
* Model versioning

---

### 🔹 4. Business-Driven Decision Layer (🔥 Critical)

Instead of default predictions:

```python
threshold = 0.3
y_pred = (y_proba > threshold).astype(int)
```

✔ Increases churn detection (recall)
✔ Aligns model with business cost structure

---

## 🧪 A/B Testing Simulation (Real-World Framing)

### 🎯 Goal:

Evaluate if the model **actually improves campaign performance**

---

### Strategies Compared:

| Strategy    | Description                     |
| ----------- | ------------------------------- |
| Baseline    | Random customer targeting       |
| Model-Based | Target high-risk customers only |

---

### 💰 Assumptions:

* Contact cost: $10
* Retention value: $200

---

### 📈 Result:

👉 Model-based strategy delivers **positive uplift in profit**
👉 Reduces unnecessary customer contact
👉 Focuses resources on high-risk segments

---

## 📊 Metrics That Matter

* ROC-AUC
* Recall (primary KPI)
* Precision
* Cross-validation stability

---

## 💼 Business Impact

✔ Increased campaign ROI
✔ Reduced operational costs
✔ Better allocation of marketing resources

👉 This model answers:

> “Who should we contact, and is it worth it?”

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

Run experiment tracking:

```bash
mlflow ui
```

---

## 🧠 What This Demonstrates

* Ability to translate ML into business value
* Understanding of cost-sensitive decision making
* Experience with experiment tracking (MLflow)
* End-to-end ML pipeline design

---

## 🔥 Next Steps (Production Vision)

* Deploy as batch scoring or API
* Integrate with CRM systems
* Run real A/B tests on campaigns
* Add SHAP for explainability (critical in banking)

---

## 👤 Profile Positioning

This project reflects the type of work expected in:

* Customer Analytics
* Risk / Retention Modeling
* Data Science for Marketing Optimization

👉 Designed for roles in banking, fintech, and data-driven organizations.

---

## 💬 Final Insight

> A good model predicts churn.
> A great model **changes business decisions**.

This project focuses on the second.
