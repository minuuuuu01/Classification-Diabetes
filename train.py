import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
# 2) X/y 분리 (분할된 데이터 기준)
X_train = train_df.drop("당뇨병 여부", axis=1).copy()
y_train = train_df["당뇨병 여부"].copy()

X_test = test_df.drop("당뇨병 여부", axis=1).copy()
y_test = test_df["당뇨병 여부"].copy()

# -------------------------------------------------------------------------
# 3) 전처리: "파라미터는 train에서만 계산" 후 train/test에 동일 적용

# (A) BMI 이상치 처리: train에서 상한선 계산 → train/test에 clip 적용
Q1 = X_train["BMI 지수"].quantile(0.25)
Q3 = X_train["BMI 지수"].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

X_train["BMI 지수"] = X_train["BMI 지수"].clip(upper=upper_limit)
X_test["BMI 지수"] = X_test["BMI 지수"].clip(upper=upper_limit)

# (B) 범주형 인코딩
# 성별: 고정 매핑
gender_map = {"Female": 0, "Male": 1, "Other": 2}
X_train["성별"] = X_train["성별"].map(gender_map)
X_test["성별"] = X_test["성별"].map(gender_map)

# 흡연 경험: 원-핫 인코딩 후, train 컬럼 기준으로 test 컬럼 정렬/보정
X_train = pd.get_dummies(X_train, columns=["흡연 경험"], prefix="흡연")
X_test = pd.get_dummies(X_test, columns=["흡연 경험"], prefix="흡연")
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# (권장) 전처리 후 안전 점검: 결측치/타입
print("X_train 결측치 총합:", X_train.isna().sum().sum())
print("X_test 결측치 총합:", X_test.isna().sum().sum())
print("object dtype columns:", X_train.select_dtypes(include="object").columns.tolist())
print("컬럼 일치:", X_train.columns.equals(X_test.columns))

# 결측치가 있다면(특히 성별 map 실패 등) 학습 전에 처리해야 함
# 여기서는 가장 단순히 결측치가 있으면 에러를 내도록 체크(원하면 대치로 바꿔드릴게요)
if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    raise ValueError("전처리 후 NaN이 남아 있습니다. 매핑 누락/결측치 대치를 먼저 처리하세요.")

if len(X_train.select_dtypes(include="object").columns) > 0:
    raise ValueError("object dtype 컬럼이 남아 있습니다. 인코딩/변환을 확인하세요.")

# -------------------------------------------------------------------------
# 4) 모델 학습(RandomForest) 및 예측

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # 클래스 불균형 가능성 있을 때 옵션(원치 않으면 제거)
)

rf.fit(X_train, y_train)  # 학습 [web:35][web:37]

y_pred = rf.predict(X_test)  # 예측 [web:35][web:37]

# -------------------------------------------------------------------------
# 5) 평가(기본)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)


print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


