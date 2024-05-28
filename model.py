
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# CSV 파일 경로
file_path = 'industry_concat.csv'
df = pd.read_csv(file_path)

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# # GRDP 및 Factor_Income 열이 NaN인 행을 삭제합니다
# df.drop(columns=['2022_c'], inplace=True)
# 변경된 데이터를 CSV 파일로 저장
# data = df
# def remove_outliers_iqr(data):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outlier_indices = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
#     filtered_data = data[~outlier_indices]
#     return filtered_data

# # 이상치 제거
# cols_to_check = ['2017_c', '2017_w', '2018_c', '2018_w', '2019_c', '2019_w',
#                  '2020_c', '2020_w', '2021_c', '2021_w', '2022_c', '2022_w',
#                  ]
# data_no_outliers = remove_outliers_iqr(data[cols_to_check])

# # 이상치 제거된 데이터셋 확인


# data_no_outliers = data.drop(columns=cols_to_check).merge(
#     remove_outliers_iqr(data[cols_to_check]), left_index=True, right_index=True
# )

# print(data_no_outliers)

# df = data_no_outliers





# # 2017년 대비 2022년 증감율 계산
df["Distribution"] = ((df["2022_w"] - df["2021_w"]) / df["2021_w"]) * 100
df.loc[df["Distribution"] == 0, "Label"] = 'zero'  # 증감없음
df.loc[df["Distribution"] < -30, "Label"] = 'big_decrease'  # 대폭감소
df.loc[(df["Distribution"] >= -30) & (df["Distribution"] < 0), "Label"] = 'small_decrease'  # 소폭감소
df.loc[(df["Distribution"] >= 0) & (df["Distribution"] < 30), "Label"] = 'small_increase'  # 소폭증가
df.loc[df["Distribution"] >= 30, "Label"] = 'big_increase'  # 대폭증가



df = df.dropna(subset=['Label'])



years = ['2017', '2018', '2019', '2020', '2021']
new_features_df = pd.DataFrame()
for i in range(1, len(years)):
    current_year = years[i]
    prev_year = years[i-1]
    
    new_feature_name = f'{current_year}_diff_{prev_year}'
    
    # 이전 연도와의 차분 계산
    new_feature_values = df[f'{current_year}_w'] - df[f'{prev_year}_w']
    
    # 새로운 변수를 데이터프레임에 추가
    new_features_df[new_feature_name] = new_feature_values

# 새로운 변수가 추가된 데이터프레임 출력
df = pd.concat([df, new_features_df], axis=1)

# 결합된 데이터프레임 출력


numeric_features = ['2017_w', '2018_w', '2019_w', '2020_w', '2021_w', '2018_diff_2017', '2019_diff_2018', '2020_diff_2019', '2021_diff_2020' ]


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])  # StandardScaler를 사용하여 표준화

# 범주형 특성에 대한 전처리 파이프라인
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One-Hot Encoding 적용

# ColumnTransformer를 사용하여 수치형과 범주형 특성에 각각 다른 전처리 적용
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ])
#('cat', categorical_transformer, categorical_features)
# 전체 데이터에 대해 전처리 수행
processed_data = preprocessor.fit_transform(df)


from sklearn.model_selection import GridSearchCV



X = processed_data
y = df['Label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# from sklearn.model_selection import GridSearchCV

# # 그리드 서치를 위한 하이퍼파라미터 그리드 설정
# param_grid = {
#     'n_estimators': [90, 95, 100],  # 트리의 개수
#     'max_depth': [16, 17, 18],  # 트리의 최대 깊이
#     'min_samples_split': [5],  # 내부 노드를 분할하는데 필요한 최소 샘플 수
#     'min_samples_leaf': [1]  # 리프 노드에 필요한 최소 샘플 수
# }

# # 그리드 서치를 위한 RandomForestClassifier 초기화
# rf_clf = RandomForestClassifier()

# # 그리드 서치 수행
# grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# rf_pred = grid_search.predict(X_test)

# # 최적의 모델과 파라미터 출력
# print("Best Parameters:", grid_search.best_params_)
# print("Best Estimator:", grid_search.best_estimator_)
# print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))




# param_grid = {
#     'n_neighbors': [8],  # 이웃의 개수
#     'weights': ['uniform', 'distance'],  # 가중치 옵션
#     'metric': ['euclidean', 'manhattan', 'chebyshev']  # 거리 측정 방법
# }

# # 그리드 서치를 위한 RandomForestClassifier 초기화
# knn_clf = KNeighborsClassifier()


# # 그리드 서치 수행
# grid_search = GridSearchCV(estimator=knn_clf, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# knn_pred = grid_search.predict(X_test)

# # 최적의 모델과 파라미터 출력
# print("Best Parameters:", grid_search.best_params_)
# print("Best Estimator:", grid_search.best_estimator_)
# print("KNN Accuracy:", accuracy_score(y_test, knn_pred))









# 분류 모델 초기화
rf_clf = RandomForestClassifier(n_estimators=90, max_depth=17, min_samples_leaf=1, min_samples_split=5)
knn_clf = KNeighborsClassifier(metric='euclidean', n_neighbors=8, weights= 'distance')

# 각 분류 모델을 학습시킵니다
rf_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)

# 테스트 데이터로 예측을 수행합니다
rf_pred = rf_clf.predict(X_test)
knn_pred = knn_clf.predict(X_test)



from sklearn.ensemble import VotingClassifier

# 보팅 분류기 초기화
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('knn', knn_clf)],
    voting='soft')  # 하드 보팅 사용

# 보팅 분류기 학습
voting_clf.fit(X_train, y_train)

# 보팅 분류기 예측
voting_pred = voting_clf.predict(X_test)
print(voting_pred)
print(df['Label'])

from sklearn.model_selection import cross_val_score

# 교차 검증을 사용하여 보팅 분류기의 성능을 평가합니다
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)

# 교차 검증 결과를 출력합니다
print("voting_Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())


# 교차 검증을 사용하여 보팅 분류기의 성능을 평가합니다
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5)

# 교차 검증 결과를 출력합니다
print("random_forest_Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())



cv_scores = cross_val_score(knn_clf, X_train, y_train, cv=5)

# 교차 검증 결과를 출력합니다
print("knn_Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())


# 각 모델의 정확도를 출력합니다
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("votting 학습",accuracy_score(y_test, voting_pred))    #votting으로 결정 





from sklearn.metrics import confusion_matrix, classification_report

# 랜덤 포레스트 모델 평가
rf_cm = confusion_matrix(y_test, rf_pred)
rf_cr = classification_report(y_test, rf_pred)

print("Random Forest Confusion Matrix:\n", rf_cm)
print("\nRandom Forest Classification Report:\n", rf_cr)

# k-최근접 이웃 모델 평가
knn_cm = confusion_matrix(y_test, knn_pred)
knn_cr = classification_report(y_test, knn_pred)

print("\nKNN Confusion Matrix:\n", knn_cm)
print("\nKNN Classification Report:\n", knn_cr)


# Vottging 모델 평가 
Votting_cm = confusion_matrix(y_test, voting_pred)
Votting_cr = classification_report(y_test, voting_pred)
class_names = ['big_decrease', 'small_decrease',  "small_increase", "big_increase"]
Votting_cr_df = pd.DataFrame(Votting_cm, index=class_names, columns=class_names)
print(Votting_cr_df)
print("\nVotting Confusion Matrix:\n", Votting_cm)
print("\nVotting Classification Report:\n", Votting_cr)





import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤 포레스트 모델의 오차 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# k-최근접 이웃 모델의 오차 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(knn_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# k-최근접 이웃 모델의 오차 행렬 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(Votting_cr_df, annot=True, fmt='d', cbar=None, cmap="Blues")
plt.title('Votting Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



##################################################2022년 데이터 예측######################################


file_path = 'industry_concat.csv'
df_goal = pd.read_csv(file_path)

# data = df_goal
# def remove_outliers_iqr(data):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outlier_indices = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
#     filtered_data = data[~outlier_indices]
#     return filtered_data

# # 이상치 제거
# cols_to_check = ['2017_c', '2017_w', '2018_c', '2018_w', '2019_c', '2019_w',
#                  '2020_c', '2020_w', '2021_c', '2021_w', '2022_c', '2022_w',
#                  ]
# data_no_outliers = remove_outliers_iqr(data[cols_to_check])

# # 이상치 제거된 데이터셋 확인


# data_no_outliers = data.drop(columns=cols_to_check).merge(
#     remove_outliers_iqr(data[cols_to_check]), left_index=True, right_index=True
# )

# print(data_no_outliers)

# df_goal = data_no_outliers




# df_goal["Distribution"] = ((df_goal["2022_w"] - df_goal["2021_w"]) / df_goal["2021_w"]) * 100
# df_goal.loc[df_goal["Distribution"] == 0, "Label"] = 'zero'  # 증감없음
# df_goal.loc[df_goal["Distribution"] < -30, "Label"] = 'big_decrease'  # 대폭감소
# df_goal.loc[(df_goal["Distribution"] >= -30) & (df_goal["Distribution"] < 0), "Label"] = 'small_decrease'  # 소폭감소
# df_goal.loc[(df_goal["Distribution"] >= 0) & (df_goal["Distribution"] < 30), "Label"] = 'small_increase'  # 소폭증가
# df_goal.loc[df_goal["Distribution"] >= 30, "Label"] = 'big_increase'  # 대폭증가



# df_goal = df_goal.dropna(subset=['Label'])
# df_goal = df_goal.dropna(subset=['Distribution'])

# print(df_goal)

years = ['2018', '2019', '2020', '2021', '2022']
new_features_df = pd.DataFrame()
for i in range(1, len(years)):
    current_year = years[i]
    prev_year = years[i-1]
    
    new_feature_name = f'{current_year}_diff_{prev_year}'
    
    # 이전 연도와의 차분 계산
    new_feature_values = df_goal[f'{current_year}_w'] - df_goal[f'{prev_year}_w']
    
    # 새로운 변수를 데이터프레임에 추가
    new_features_df[new_feature_name] = new_feature_values

# 새로운 변수가 추가된 데이터프레임 출력
df_goal = pd.concat([df_goal, new_features_df], axis=1)
# df_goal = df_goal.drop(columns=['Distribution', 'Label'])

print(df_goal)
numeric_features = ['2018_w', '2019_w', '2020_w', '2021_w', '2022_w', '2019_diff_2018', '2020_diff_2019', '2021_diff_2020', '2022_diff_2021' ]


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])  # StandardScaler를 사용하여 표준화

# 범주형 특성에 대한 전처리 파이프라인
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One-Hot Encoding 적용

# ColumnTransformer를 사용하여 수치형과 범주형 특성에 각각 다른 전처리 적용
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ])
#('cat', categorical_transformer, categorical_features)
# 전체 데이터에 대해 전처리 수행
processed_data = preprocessor.fit_transform(df_goal)
print(processed_data)
labeling = voting_clf.predict(processed_data)
df_goal['2023_predict'] = labeling
print(df_goal)
df_goal.to_csv("df_goal.csv", index=False)
