# 선형회귀 모델 추세식 계산
import tensorflow as tf
import numpy as np

class LinearRegressionTest:
    def __init__(self, learning_rate, epochs):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x:np.ndarray, y:np.ndarray):
        # 경사하강법(Gradient Descent)으로 w, b를 학습
        # parameter 초기화
        self.w = np.random.uniform(-2, 2)
        self.b = np.random.uniform(-2, 2)

        n = len(x) # 데이터 건수

        for epoch in range(self.epochs):
            y_pred = self.w * x + self.b # 예측값 계산
            loss = np.mean((y - y_pred) ** 2) # 손실

            dw = (-2 / n) * np.sum(x * (y - y_pred)) # 경사 계산. 편미분
            db = (-2 / n) * np.sum(y - y_pred) # 경사 계산. 편미분

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db   

            # 학습 상태 출력
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {loss:.5f}, w: {self.w:.5f}, b: {self.b:.5f}')

    def predict(self, x:np.ndarray):
        return self.w * x + self.b
    
def main():
    np.random.seed(42)

    # feature
    x_heights = np.random.normal(175, 10, 30)
    
    true_w = 0.7
    true_b = -55
    noise = np.random.normal(0, 3, 30)
    
    # label
    y_weights = true_w * x_heights + true_b + noise
    print(x_heights)
    print(y_weights)

    # 스케일링
    x_mean = np.mean(x_heights)
    x_std = np.std(x_heights)
    y_mean = np.mean(y_weights)
    y_std = np.std(y_weights)

    x_heights_scaled = (x_heights - x_mean) / x_std
    y_weights_scaled = (y_weights - y_mean) / y_std

    # 모델 학습
    model = LinearRegressionTest(learning_rate=0.01, epochs=100)
    model.fit(x_heights_scaled, y_weights_scaled)

    print(f'Learned parameters - w: {model.w}, b: {model.b}')

    # 예측
    y_pred_scaled = model.predict(x_heights_scaled)

    # 예측 결과 역변환
    y_pred = (y_pred_scaled * y_std) + y_mean + y_mean
    print('y_pred:', y_pred)

    # 모델 성능 (MSE, R^2) 계산
    mse = np.mean((y_weights - y_pred) ** 2)
    ss_tot = np.sum((y_weights - np.mean(y_weights)) ** 2)

    ss_res = np.sum((y_weights - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print('학습결과 -------------------------')
    print('추정된 기울기 w :', model.w)
    print('추정된 b :', model.b)

    for i in range(len(x_heights)):
        print(f'키:{x_heights[i]:.2f}cm, 몸무게 실제:{y_weights[i]:.2f}kg, 예측:{y_pred[i]:.2f}kg')

    print(f'mse : {mse:.4f}, r^2 : {r2:.4f}')
    print(f'R^2 : {mse:.4f}, r^2 : {r2:.4f}')


if __name__ == '__main__':
    main()