# pcaKnnTest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
data = load_iris()

iris_df = pd.DataFrame(data['data'], columns=data['feature_names'])
iris_df['target'] = data['target']
print(iris_df.head())
print(iris_df.describe())
iris_df.info()
print(iris_df['target'].value_counts())

# 전체 데이터 셋에 대한 히스토그램
# iris_df.drop('target', axis=1).hist(figsize=(10, 8), bins=20, edgecolor='black')
# plt.suptitle("Histogram of Iris Dataset")
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 품종 이름 매핑
iris_df['species'] = iris_df['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})
print(iris_df[['target', 'species']])

# # species(품종 이름)를 기준으로 히스토그램 그리기
# feature_list = iris_df.columns[:-2]  # feature만 골라오기 (target, species 제외)

# for feature in feature_list:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(data=iris_df, x=feature, hue='species', bins=20, kde=True, palette='Set2')
#     plt.title(f'Histogram of {feature} by Species')
#     plt.xlabel(feature)
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# 특성과 라벨 분리
X = iris_df.drop(['target', 'species'], axis=1)
y = iris_df['target']

# 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

pca = PCA()
pca.fit(X_scaled)
print("주성분:", pca.components_)
print("주성분 분산:", pca.explained_variance_)
print("주성분 분산 비율:", pca.explained_variance_ratio_)

# 주성분 로딩 (고유백터)
loadings = pca.components_
print("Loadings:\n", loadings)

# 탐색적 분석
# plt.figure(figsize=(8, 8))
# ax = sns.pairplot(iris_df, hue='target')
# plt.grid(True)
# plt.show()

# plt.plot(pca.explained_variance_, marker='o')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue')
# plt.grid(True)
# plt.show()

# 주성분으로 데이터 변환
Z = pca.transform(X_scaled)
print(Z)
# 데이터프레임으로 변환
Z_df = pd.DataFrame(data=Z, columns=[f'PC{i+1}' for i in range(Z.shape[1])])
print(Z_df.head())

# Z_df['target'] = data['target']
# plt.figure(figsize=(8, 8))
# ax = sns.pairplot(Z_df, hue='target')
# plt.grid(True)
# plt.show()

loadings = pca.components_
print("Loadings:\n", loadings)

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

#  정확도 계산 함수
def iris_pca_knn(X, y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

#  K=1~15에 대한 정확도 측정
accuracies_pca = []
k_range = range(1, 16)

for k in k_range:
    acc = iris_pca_knn(X_pca_2d, y, k)
    accuracies_pca.append(acc)
    print(f"K={k} 정확도: {acc:.4f}")

#  정확도 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies_pca, marker='o', linestyle='-', color='teal')
plt.title('K 값에 따른 KNN 정확도')
plt.xlabel('K 값')
plt.ylabel('정확도')
plt.xticks(k_range)
plt.grid(True)
plt.show()