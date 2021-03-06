# 20-06-08_20
# 월요일 // 12:30 ~

# cnn 구성


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D
import numpy as np


''' 1. 데이터 '''

# data 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # 60000, 28, 28
print(x_test.shape)     # 10000, 28, 28

# data 전처리, 리쉐이프
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')/255
print(x_train.shape)    # (60000, 28, 28, 1)
print(x_test.shape)     # (10000, 28, 28, 1)

# y 원핫인코딩 (차원 확인할 것)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)    # 60000, 10
print(y_test.shape)     # 10000, 10


''' 2. 모델 '''

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(80, (2,2), activation='relu', padding='same', name='hiddencv1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(80, (2,2), activation='relu', padding='same', name='hiddencv2')(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(80, (2,2), activation='relu', padding='same', name='hiddencv3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
print('최적의 매개변수 :', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어 :', acc)

# pinrt 값 (node : 80, 80, 80)
# 최적의 매개변수 : {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 20}
# 최종 스코어 : 0.9846000075340271