# 붓꽃(Iris) 데이터 K-NN 및 PCA 성능 비교

이 프로젝트는 붓꽃(Iris) 데이터셋을 활용하여, 원본 데이터를 사용한 K-NN 모델과 PCA로 차원을 축소한 데이터를 사용한 K-NN 모델의 성능을 비교 분석합니다.

## 1. 프로젝트 목표

* 붓꽃 데이터셋의 4가지 원본 특성(sepal length/width, petal length/width)을 모두 사용하여 K-NN 모델의 성능을 평가합니다.
* 주성분 분석(PCA)을 적용하여 데이터의 차원을 2개로 축소합니다.
* 2개의 주성분(PC1, PC2)만 사용하여 K-NN 모델의 성능을 평가하고 원본 모델과 비교합니다.

## 2. 사용된 기술

* **Python**
* **scikit-learn**:
    * 데이터셋 로드 (`load_iris`)
    * 데이터 분리 (`train_test_split`)
    * 데이터 스케일링 (`StandardScaler`)
    * 차원 축소 (`PCA`)
    * 모델 학습 (`KNeighborsClassifier`)
    * 성능 평가 (`accuracy_score`)
* **pandas**: 데이터프레임 변환 및 조작
* **seaborn** & **matplotlib**: 데이터 시각화 (Pairplot, Scatter plot)

## 3. 실험 과정 (`main.py`)

1.  **데이터 로드 및 탐색 (EDA)**
    * `load_iris()` 함수로 붓꽃 데이터를 불러옵니다.
    * `pandas` DataFrame으로 변환 후, `seaborn.pairplot`을 사용하여 4개 특성 간의 관계와 품종별 분포를 시각화합니다.

    
    <img width="828" height="847" alt="image" src="https://github.com/user-attachments/assets/43fbb8dc-28d9-4217-b797-d3f14eadc761" />
    ![<붓꽃 데이터 Pairplot>]
    <img width="783" height="691" alt="image" src="https://github.com/user-attachments/assets/da1e8e51-1d9f-47f7-b88f-a60c855c6482" />
    ![<Iris 데이터 셋 전체에 대한 히스토그램>]

    <img width="470" height="378" alt="image" src="https://github.com/user-attachments/assets/619d9101-23f4-41de-83f7-cd633a2210c7" />
    <img width="470" height="378" alt="image" src="https://github.com/user-attachments/assets/0579e3d6-65a3-4e2d-b45c-b987e1da463b" />
    <img width="470" height="378" alt="image" src="https://github.com/user-attachments/assets/7ca13647-4e70-45dd-a43d-dcaf631dfcf8" />
    <img width="470" height="378" alt="image" src="https://github.com/user-attachments/assets/4cd3c88d-abf5-4d8c-a2c5-c063399fbdd8" />


3.  **실험 1: 원본 데이터 K-NN 성능 평가**
    * 원본 데이터를 훈련(70%) 및 테스트(30%) 세트로 분리합니다 (`test_size=0.3`, `random_state=42`).
    * `StandardScaler`를 훈련 세트에 `fit_transform`하고, 테스트 세트에는 `transform`하여 데이터 스케일을 표준화합니다.
    * `K` 값을 1부터 15까지 변경해가며 K-NN 모델을 학습시키고, 각 `K` 값에 따른 테스트 정확도를 측정하여 출력합니다.

4.  **실험 2: PCA 적용 후 K-NN 성능 평가**
    * **전체** 원본 데이터에 `StandardScaler`를 적용하여 스케일링합니다.
    * `PCA(n_components=2)`를 사용하여 스케일링된 4차원 데이터를 2차원(주성분 2개)으로 축소합니다.
    * PCA로 변환된 2차원 데이터를 훈련(70%) 및 테스트(30%) 세트로 분리합니다 (`random_state=42`).
    * `K` 값을 1부터 15까지 변경해가며 K-NN 모델을 학습시키고, 각 `K` 값에 따른 테스트 정확도를 측정하여 출력합니다.

5.  **PCA 결과 시각화**
    * 2개의 주성분(PC1, PC2)으로 변환된 붓꽃 데이터 전체를 `matplotlib`의 `scatter` 함수를 이용해 2D 평면에 시각화합니다.
    * 각 점은 실제 붓꽃의 품종(target)에 따라 다른 색상으로 구분하여 표시됩니다.

    <img width="783" height="534" alt="image" src="https://github.com/user-attachments/assets/1c093f79-1be5-4292-a8e4-b7fd77bffd2c" />
    ![붓꽃 데이터를 원본 데이터 그대로 사용한 K-NN]
    
    <img width="783" height="534" alt="image" src="https://github.com/user-attachments/assets/61243f52-898f-4106-a3a5-df46ff91bacb" />
    ![PCA로 적용한 2개의 차원만 사용한 K-NN의 성능]


## 4. 파일 구성

* `main.py`: 모든 실험과 시각화 코드가 포함된 메인 Python 스크립트입니다.
* `과제1_머신러닝기초_2401110268_차기환.pdf`: 프로젝트의 요구사항과 분석 내용이 담긴 과제 보고서입니다.

## 5. 실행 방법

1.  **필요한 라이브러리 설치:**
    ```bash
    pip install scikit-learn pandas seaborn matplotlib
    ```

2.  **스크립트 실행:**
    ```bash
    python main.py
    ```
    * 실행 시 콘솔에 두 가지 실험(원본, PCA)의 K값(1~15)에 따른 정확도가 각각 출력됩니다.
    * Pairplot과 PCA 2D Scatter plot이 팝업 창으로 시각화됩니다.
