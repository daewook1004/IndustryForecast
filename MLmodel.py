
"""
Industry Workforce Change Prediction Pipeline
--------------------------------------------
본 스크립트는 2017~2022년 산업별 종사자 수 데이터를 기반으로
산업의 증감 패턴을 분류하고, 2023년의 산업 변화를 예측하는 파이프라인입니다.
"""

# -----------------------------
# 1. 라이브러리 임포트
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# 2. 데이터 불러오기
# -----------------------------
file_path = 'dataset/industry_concat.csv'
df = pd.read_csv(file_path)


# -----------------------------
# 3. 라벨링: 2017~2022년 변화율 기반 증감 카테고리 생성
# -----------------------------
df["Distribution"] = ((df["2022_w"] - df["2021_w"]) / df["2021_w"]) * 100

df.loc[df["Distribution"] == 0, "Label"] = 'zero'
df.loc[df["Distribution"] < -30, "Label"] = 'big_decrease'
df.loc[(df["Distribution"] >= -30) & (df["Distribution"] < 0), "Label"] = 'small_decrease'
df.loc[(df["Distribution"] >= 0) & (df["Distribution"] < 30), "Label"] = 'small_increase'
df.loc[df["Distribution"] >= 30, "Label"] = 'big_increase'

df = df.dropna(subset=['Label'])

# -----------------------------
# 4. 연도별 차분 피처 생성 (Feature Engineering)
# -----------------------------
years = ['2017', '2018', '2019', '2020', '2021']
for i in range(1, len(years)):
    current_year = years[i]
    prev_year = years[i-1]
    df[f'{current_year}_diff_{prev_year}'] = df[f'{current_year}_w'] - df[f'{prev_year}_w']

# -----------------------------
# 5. 전처리 설정 (Scaling)
# -----------------------------
numeric_features = [
    '2017_w', '2018_w', '2019_w', '2020_w', '2021_w',
    '2018_diff_2017', '2019_diff_2018', '2020_diff_2019', '2021_diff_2020'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_features),
    ])

# -----------------------------
# 6. 데이터 분할
# -----------------------------
X = preprocessor.fit_transform(df)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# -----------------------------
# 7. 분류 모델 정의 및 학습
# -----------------------------
rf_clf = RandomForestClassifier(n_estimators=90, max_depth=17, min_samples_leaf=1, min_samples_split=5)
knn_clf = KNeighborsClassifier(metric='euclidean', n_neighbors=8, weights='distance')

rf_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)

# 보팅 분류기 (Soft Voting)
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('knn', knn_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# -----------------------------
# 8. 모델 예측 및 평가
# -----------------------------
rf_pred = rf_clf.predict(X_test)
knn_pred = knn_clf.predict(X_test)
voting_pred = voting_clf.predict(X_test)

# 정확도 비교
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("Voting Accuracy:", accuracy_score(y_test, voting_pred))

# 교차 검증
for name, model in [('Voting', voting_clf), ('RandomForest', rf_clf), ('KNN', knn_clf)]:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} Cross-validation Mean: {cv_scores.mean():.3f}")

# -----------------------------
# 9. 혼동 행렬 및 분류 리포트
# -----------------------------
models = {
    "RandomForest": rf_pred,
    "KNN": knn_pred,
    "Voting": voting_pred
}

for name, pred in models.items():
    print(f"\n{name} Classification Report:\n", classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# -----------------------------
# 10. 2023년 예측
# -----------------------------
df_goal = pd.read_csv(file_path)

# 연도별 차분 추가 (2022까지)
years_goal = ['2017', '2018', '2019', '2020', '2021', '2022']
for i in range(1, len(years_goal)):
    current_year = years_goal[i]
    prev_year = years_goal[i-1]
    df_goal[f'{current_year}_diff_{prev_year}'] = df_goal[f'{current_year}_w'] - df_goal[f'{prev_year}_w']


# 학습 피처와 동일한 구조로 설정
numeric_features_goal = [
    '2017_w', '2018_w', '2019_w', '2020_w', '2021_w',
    '2018_diff_2017', '2019_diff_2018', '2020_diff_2019', '2021_diff_2020'
]

preprocessor_goal = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]), numeric_features_goal),
    ])

X_goal = preprocessor_goal.fit_transform(df_goal)
df_goal['2023_predict'] = voting_clf.predict(X_goal)

# 결과 저장
df_goal.to_csv("df_goal.csv", index=False)
print("\n✅ 2023 예측 결과가 'df_goal.csv' 파일로 저장되었습니다.")



