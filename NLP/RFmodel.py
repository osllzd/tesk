from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import pandas as pd

class RFModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier()
        self.is_trained = False

    def fit(self, X_train, y_train):
        # 모델 훈련
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        predictions = self.model.predict(X)
        class_labels = ['ballad' if label == 0 else 'dance' for label in predictions]
        return class_labels

        # 예측

    def predict_proba(self, X):

        # 클래스에 대한 예측 확률
        return self.model.predict_proba(X)



    def evaluate(self, X,y):

        # 예측
        predictions = self.model.predict(X)
        # 정확도 평가
        accuracy = accuracy_score(y, predictions)
        # 손실률 평가
        loss = log_loss(y, predictions)
        return accuracy, loss

# 데이터프레임 로드 및 전처리
kpopDF = pd.read_csv('../datasets/kpop.csv')
kpopDF.index = ['balad' + str(i) for i in range(1, 101)] + ['dance' + str(i) for i in range(1, 101)]
kpopDF['Label'] = kpopDF.index.to_series().apply(lambda x: 0 if str(x).startswith('balad') else 1)
kpopDF=kpopDF.drop('Filename',axis=1)
targetDF = kpopDF['Label']
featureDF = kpopDF.drop(columns=['Label'])
X_train, X_test, y_train, y_test = train_test_split(featureDF, targetDF, test_size=0.2, random_state=42)

# 모델 생성
model = RFModel()

# 모델 훈련
model.fit(X_train, y_train)


# 정확도와 손실률 평가
accuracy, loss = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)
print("Loss:", loss)

# 사용자로부터 입력 받기
input_value = float(input("수 입력: "))

# 기존 DataFrame에 입력값을 추가하여 예측을 수행
input_feature = featureDF.copy()  # 기존 DataFrame 복사
input_feature['Chromagram_Mean'] = input_value  # 새로운 입력값 추가

# 모델로 예측하기
prediction = model.predict(input_feature)

# 예측 결과 출력
print("예측 결과:", prediction[0])
