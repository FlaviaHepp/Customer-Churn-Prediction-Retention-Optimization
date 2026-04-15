"""Acerca de este conjunto de datos
Este conjunto de datos contiene una gran cantidad de información de clientes recopilada dentro de una cartera de tarjetas de crédito de consumo, con el objetivo de ayudar a los analistas a predecir la pérdida de clientes. Incluye detalles demográficos completos como edad, género, estado civil y categoría de ingresos, así como información sobre la relación de cada cliente con el proveedor de la tarjeta de crédito, como el tipo de tarjeta, el número de meses de antigüedad y los períodos de inactividad. Además, contiene datos clave sobre el comportamiento de gasto de los clientes a medida que se acercan a su decisión de darse de baja, como el saldo rotatorio total, el límite de crédito, la tasa promedio de apertura a compra y métricas analizables como el monto total de cambio del cuarto trimestre al primer trimestre, la tasa de utilización promedio y el indicador de abandono del clasificador Naive Bayes (la categoría de la tarjeta se combina con el número de contactos en un período de 12 meses junto con el número de dependientes, el nivel educativo y los meses de inactividad). Ante este conjunto de puntos de datos predictivos útiles a través de múltiples variables, se captura información actualizada que puede determinar la estabilidad de la cuenta a largo plazo o una inminente baja, lo que nos brinda una comprensión más completa al buscar administrar una cartera o atender a clientes individuales."""

#Metadata
#Column name ==>Description
#CLIENTNUM ==>Unique identifier for each customer. (Integer)
#Attrition_Flag ==>Flag indicating whether or not the customer has churned out. (Boolean)
#Customer_Age ==>Age of customer. (Integer)
#Gender ==>Gender of customer. (String)
#Dependent_count ==>Number of dependents that customer has. (Integer)
#Education_Level ==>Education level of customer. (String)
#Marital_Status ==>Marital status of customer. (String)
#Income_Category ==>Income category of customer. (String)
#Card_Category ==>Type of card held by customer. (String)
#Months_on_book ==>How long customer has been on the books. (Integer)
#Total_Relationship_Count ==>Total number of relationships customer has with the credit card provider. (Integer)
#Months_Inactive_12_mon ==>Number of months customer has been inactive in the last twelve months. (Integer)
#Contacts_Count_12_mon ==>Number of contacts customer has had in the last twelve months. (Integer)
#Credit_Limit ==>Credit limit of customer. (Integer)
#Total_Revolving_Bal ==>Total revolving balance of customer. (Integer)
#Avg_Open_To_Buy ==>Average open to buy ratio of customer. (Integer)
#Total_Amt_Chng_Q4_Q1 ==>Total amount changed from quarter 4 to quarter 1. (Integer)
#Total_Trans_Amt ==>Total transaction amount. (Integer)
#Total_Trans_Ct ==>Total transaction count. (Integer)
#Total_Ct_Chng_Q4_Q1 ==>Total count changed from quarter 4 to quarter 1. (Integer)
#Avg_Utilization_Ratio ==>Average utilization ratio of customer. (Integer)
#Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 ==>
#Naive Bayes classifier for predicting whether or not someone will churn based on characteristics such

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("BankChurners.csv")

# Target
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

X = df.drop(columns=["Attrition_Flag", "CLIENTNUM"])
y = df["Attrition_Flag"]

# =========================
# 2. FEATURE TYPES
# =========================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# =========================
# 3. PREPROCESSING
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# =========================
# 4. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# =========================
# 5. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 6. TRAIN
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 7. EVALUATION
# =========================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# =========================
# MLflow experiment
# =========================
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():

    # =========================
    # MODEL BASE
    # =========================
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_split": [2, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1
    )

    # =========================
    # TRAIN
    # =========================
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # =========================
    # EVAL
    # =========================
    y_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Best Params:", search.best_params_)
    print("ROC-AUC:", roc_auc)

    # =========================
    # LOGGING 🔥
    # =========================
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(best_model, "model")
    
    # ejemplo simple
threshold = 0.3  # más recall

y_custom = (y_proba > threshold).astype(int)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

mlflow.log_metric("cv_mean_auc", cv_scores.mean())
mlflow.log_metric("cv_std_auc", cv_scores.std())

#Simulación A/B Testing

import numpy as np

cost_contact = 10        # costo de contactar cliente
benefit_retention = 200  # beneficio si cliente se queda

# Probabilidades del modelo
y_proba = best_model.predict_proba(X_test)[:, 1]

# Estrategia A (random)
random_group = np.random.choice([0, 1], size=len(y_test), p=[0.7, 0.3])

# Estrategia B (modelo)
threshold = 0.3
model_group = (y_proba > threshold).astype(int)

def calculate_profit(y_true, selected, cost, benefit):
    
    profit = 0
    
    for i in range(len(y_true)):
        if selected[i] == 1:  # cliente contactado
            profit -= cost
            
            if y_true[i] == 1:  # cliente realmente iba a churn
                profit += benefit
    
    return profit

profit_random = calculate_profit(y_test.values, random_group, cost_contact, benefit_retention)
profit_model = calculate_profit(y_test.values, model_group, cost_contact, benefit_retention)

print("Profit Random:", profit_random)
print("Profit Model:", profit_model)

uplift = profit_model - profit_random
print("Uplift generado por el modelo:", uplift)

