# iris 다항 분류
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
print(type(iris))  # <class 'sklearn.utils._bunch.Bunch'>
print(iris.keys())
x = iris.data
y = iris.target
print(x[:3])
print(y[:3])
print(x.shape, y.shape) # (150, 4) (150,)
print(set(y))

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label : onehot
onehot = OneHotEncoder(categories='auto') # to_categorical, numpy:np.eye(), pd.get_dummies()
print(y.shape) # (150,)
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y.shape) # (150, 3)
print(y[:3])

# feature: 표준화

print(x[:2])
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3)

# shape에서 int로 추출해야 함
n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print(n_features, n_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_model_func(input_dim, out_dim, out_nodes, n, model_name = 'model'):
    # print(input_dim, out_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name=model_name)
        model.add(Input(shape=(input_dim,)))
        for _ in range(n):
            model.add(Dense(10, activation='relu'))
        model.add(Dense(out_dim, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    return create_model  # 이 줄 추가!
models = [create_model_func(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1, 4)]
print(len(models))

for create_model in models:
    print()
    create_model().summary()

history_dict = {}
for create_model in models:
    model = create_model()
    print('모델명 :', model.name)
    history = model.fit(x_train, y_train, batch_size=8, epochs=50, verbose=0, validation_split=0.2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('test dataset loss: ', score[0])
    print('test dataset accuracy: ', score[1])
    history_dict[model.name] = [history, model]

print(history_dict)


# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['accuracy'])
    val_acc = history_dict[model_name][0].history['val_accuracy']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('val_accuracy')
    ax2.set_ylabel('val_loss')
    ax2.set_xlabel('epoch')
    ax1.legend()
    ax2.legend()
plt.savefig('./deeplearning/data/user_data/images/25_roc_1.png')

# ROC Curve : 분류기에 대한 성능 평가 방벙중 하나
from sklearn.metrics import roc_curve, auc


plt.figure()
plt.plot([0, 1], [0,1], 'k--')

for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(x_test)
    # Convert sparse matrices to dense arrays and then ravel
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label = '{}, auc value : {:.3f}'.format(model_name, auc(fpr, tpr)))

plt.xlabel('False Positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend()

plt.savefig('./deeplearning/data/user_data/images/25_roc_2.png')
# plt.show()