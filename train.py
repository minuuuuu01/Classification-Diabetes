import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report


# 데이터 로드
df = pd.read_csv("diabetes_prediction_dataset.csv", encoding="cp949")

# (선택) 기본 확인/EDA
df.info()
print(df.describe())


# -------------------------------------------------------------------------
# 1) 데이터 분할(8:2) - 전처리(특히 통계치 계산/규칙 학습) 전에 먼저 수행
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["당뇨병 여부"]
)


# -------------------------------------------------------------------------
# 2) 전처리를 "훈련 폴드에만 fit"해서 적용하기 위한 함수
def preprocess_train_valid(X_tr: pd.DataFrame, X_va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    # (A) BMI 이상치 처리: train(fold)에서 상한선 계산 → train/valid에 동일 적용
    Q1 = X_tr["BMI 지수"].quantile(0.25)
    Q3 = X_tr["BMI 지수"].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR

    X_tr["BMI 지수"] = X_tr["BMI 지수"].clip(upper=upper_limit)
    X_va["BMI 지수"] = X_va["BMI 지수"].clip(upper=upper_limit)

    # (B) 범주형 인코딩
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    X_tr["성별"] = X_tr["성별"].map(gender_map)
    X_va["성별"] = X_va["성별"].map(gender_map)

    # 흡연 경험: 원-핫 인코딩 후, train 컬럼 기준으로 valid 컬럼 정렬/보정
    X_tr = pd.get_dummies(X_tr, columns=["흡연 경험"], prefix="흡연")
    X_va = pd.get_dummies(X_va, columns=["흡연 경험"], prefix="흡연")
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

    # (권장) 전처리 후 안전 점검
    if X_tr.isna().sum().sum() > 0 or X_va.isna().sum().sum() > 0:
        raise ValueError("전처리 후 결측치가 발생했습니다. (예: 성별/흡연 값 매핑 실패)")

    return X_tr, X_va


# -------------------------------------------------------------------------
# 3) 기본 KFold 교차 검증(훈련 데이터 내부에서만) - 성능지표: macro F1
X_all = train_df.drop("당뇨병 여부", axis=1).copy()
y_all = train_df["당뇨병 여부"].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # KFold 옵션 [web:96]

cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), start=1):
    X_tr_raw = X_all.iloc[tr_idx]
    y_tr = y_all.iloc[tr_idx]
    X_va_raw = X_all.iloc[va_idx]
    y_va = y_all.iloc[va_idx]

    X_tr, X_va = preprocess_train_valid(X_tr_raw, X_va_raw)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X_tr, y_tr)

    y_va_pred = rf.predict(X_va)
    f1_macro = f1_score(y_va, y_va_pred, average="macro")  # macro F1 정의 [web:109]
    cv_scores.append(f1_macro)

    print(f"[Fold {fold}] valid_size={len(va_idx)}, macro_f1={f1_macro:.4f}")

print("\n[KFold CV 결과 - macro F1]")
print(f"macro_f1_mean={np.mean(cv_scores):.4f}, macro_f1_std={np.std(cv_scores):.4f}")


# -------------------------------------------------------------------------
# 4) 홀드아웃 테스트 평가: train_df로 fit, test_df로 최종 점검
X_train_raw = train_df.drop("당뇨병 여부", axis=1).copy()
y_train = train_df["당뇨병 여부"].copy()

X_test_raw = test_df.drop("당뇨병 여부", axis=1).copy()
y_test = test_df["당뇨병 여부"].copy()

X_train, X_test = preprocess_train_valid(X_train_raw, X_test_raw)

rf_final = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf_final.fit(X_train, y_train)

y_test_pred = rf_final.predict(X_test)

print("\n[Hold-out Test Classification Report]\n", classification_report(y_test, y_test_pred, digits=4))  # 리포트에 macro avg 포함 [web:118]
print("[Hold-out Test macro F1]", f1_score(y_test, y_test_pred, average="macro"))  # macro F1 계산 [web:109]
