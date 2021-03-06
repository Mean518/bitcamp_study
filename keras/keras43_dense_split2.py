# 42번을 카피하여 Dense로 리뉴얼 !!!
# 42와 비교하여 성능을 더 높이시오 !!!

'''
정말 LSTM 이랑 
성능비교 자체가 안되는구만
mse 는 8.44011111000853e-11
[[94.99999]
 [95.99999]
 [96.99999]
 [97.99999]
 [98.99999]
 [99.99999]]
RMSE 는 7.62939453125e-06
R2는  0.9999999999800431
'''

import numpy as np

a = np.array(range(1,101))
size = 5
def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa)) 
    return np.array(aaa)

dataset = split_x(a, size)
x = np.array(dataset[:,:4])
y = np.array(dataset[:,4])
x_predict = x[-6:,:]
x_predict_y = y[-6:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.20, shuffle=False
)

#2. 모델구성
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

model = Sequential()
model.add(Dense(32, input_dim=(4))) # 마지막 input을 넣는거야 무조권 ^_^ -> 1의 자리 숫자..!!!
model.add(Dense(32))
model.add(Dense(640))
# model.add(Dense(320))
model.add(Dense(16))  
model.add(Dense(1)) 

#3. 설명한 후 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') # 어느정도 사이즈부터 성능향상에 도움을 주는지 아직 머르겠씁니다
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=200, batch_size=5, callbacks=[early_stopping], validation_split=0.2)  

#4. 평가와 예측
loss,mse = model.evaluate(x_test,y_test) 
print('mse 는',mse)

x_predict = model.predict(x_predict)
print(x_predict)

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE 는', RMSE(x_predict_y, x_predict) )

# R2 구하기
from sklearn.metrics import r2_score 
r2 = r2_score(x_predict_y, x_predict)
print('R2는 ', r2)
