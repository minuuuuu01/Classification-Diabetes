import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


# -----------------------------
# 0) 데이터 로드
df = pd.read_csv("diabetes_prediction_dataset.csv", encoding="cp949")
df.info()
print(df.describe())


# -----------------------------
# 1) Hold-out split (8:2)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["당뇨병 여부"]
)

target_col = "당뇨병 여부"
X_train_raw = train_df.drop(target_col, axis=1).copy()
y_train = train_df[target_col].copy()
X_test_raw = test_df.drop(target_col, axis=1).copy()
y_test = test_df[target_col].copy()


# -----------------------------
# 2) 전처리(폴드별 fit/transform)

class BMIClipper(BaseEstimator, TransformerMixin):
    # 목적: BMI 이상치 상한 clip (상한은 fold의 train에서만 계산 → 누수 방지)
    def __init__(self, bmi_index: int):
        self.bmi_index = bmi_index

    def fit(self, X, y=None):
        X = np.asarray(X)
        bmi = X[:, self.bmi_index].astype(float)
        q1 = np.quantile(bmi, 0.25)
        q3 = np.quantile(bmi, 0.75)
        iqr = q3 - q1
        self.upper_limit_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.asarray(X).copy()
        bmi = X[:, self.bmi_index].astype(float)
        X[:, self.bmi_index] = np.minimum(bmi, self.upper_limit_)
        return X


cat_cols = ["성별", "흡연 경험"]
num_cols = ["나이", "고혈압 여부", "심장질환 여부", "BMI 지수", "당화혈색소 수치", "혈당 수치"]
bmi_idx_in_num = num_cols.index("BMI 지수")

numeric_pipe = Pipeline(steps=[
    ("bmi_clip", BMIClipper(bmi_index=bmi_idx_in_num)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)


# -----------------------------
# 3) 모델 + GridSearchCV (cv=5 고정, CPU 친화 설정)

# 중첩 병렬 방지:
# - GridSearchCV가 병렬로 여러 조합/폴드를 돌릴 수 있으니(n_jobs),
# - RandomForest 내부 병렬(n_jobs)은 1로 둬서 “과도한 스레드 폭발”을 막는다. [web:81][web:289]
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=1,              # 중요: RF 내부 병렬 OFF (중첩 병렬 방지)
        bootstrap=True
    ))
])

# 속도 최우선: 영향 큰 파라미터만 “소수 후보”로 제한
param_grid = {
    # n_estimators: 트리 개수(시간에 가장 큰 영향). 너무 넓게 잡으면 바로 폭발하니 3개만. [web:235]
    "model__n_estimators": [120, 200, 300],

    # max_depth: 과적합/표현력 조절. None 포함 + 얕은 값 1개만 추가(2개만). 
    "model__max_depth": [None, 12],

    # min_samples_leaf: 일반화에 자주 영향. 후보 2개만.
    "model__min_samples_leaf": [1, 5],

    # class_weight: 불균형 보정 여부만 비교(2개만).
    "model__class_weight": [None, "balanced"],

    # max_features: 이 데이터는 피처 수가 많지 않아서 'sqrt' 고정해 조합 수를 줄임(속도 목적).
    "model__max_features": ["sqrt"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # cv=5 고정 [web:97]

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,          # GridSearch 쪽만 병렬 사용
    pre_dispatch=2,     # 동시에 띄우는 작업 수 제한(메모리/멈춤 완화에 도움될 수 있음) [web:294]
    verbose=2,
    refit=True,
    return_train_score=False
)

grid.fit(X_train_raw, y_train)

print("\n[GridSearchCV Best]")
print("best_params_:", grid.best_params_)
print(f"best_cv_macro_f1: {grid.best_score_:.4f}")


# -----------------------------
# 4) Hold-out test 평가
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test_raw)

print("\n[Hold-out Test Classification Report]\n", classification_report(y_test, y_test_pred, digits=4))
print("[Hold-out Test macro F1]", f1_score(y_test, y_test_pred, average="macro"))
