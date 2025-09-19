df = pd.read_csv('student-mat.csv', sep=';')
X = df[['age', 'studytime', 'failures', 'absences']]  # 예시
y = df['G3']
X_train, X_test, y_train, y_test = 1:_____________________(X, y, test_size=0.2, random_state=42)  # train_test_split

# 정규화
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = 2:______________________________    # [빈칸 2] 테스트 데이터 변환



# 모델 구성
model = 3:________________          # [빈칸 3] 순차적 층(Layer) 구성
# (Input(shape=(4, )

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(4:________________))    # [빈칸 4] 출력층 구성
# model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

y_pred = 5:___________________      # [빈칸 5] 테스트 데이터에 대한 예측