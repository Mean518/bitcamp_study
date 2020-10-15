from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

dataset = load_boston()

x = dataset.data
y = dataset.target
# 전처리
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(
    x, y, random_state=66, test_size=0.2 )
#----------------------------------------------------------
model = KNeighborsRegressor()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_predict,y_test)
r2 = r2_score(y_predict,y_test)
# print('mse는',mse)
print('r2_1는',r2)
score = model.score(x_test,y_test) # 회귀(r2가 나옴)던 분류(acc가 나옴)던 사용할 수 있음
print(score)
#----------------------------------------------------------
model = RandomForestRegressor()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
mse = mean_squared_error(y_predict,y_test)

r2 = r2_score(y_predict,y_test)
print('mse는',mse)
print('r2_2는',r2)
score = model.score(x_test,y_test) # 회귀(r2가 나옴)던 분류(acc가 나옴)던 사용할 수 있음
print(score)


