import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

df  = pd.read_csv("data/demand/retail_sales.csv", parse_dates=['date'])
df = df.sort_values(['store','item','date'])

df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year


df['lag_1'] = df.groupby(['store','item'])['sales'].shift(1)
df['lag_7'] = df.groupby(['store','item'])['sales'].shift(7)
df["lag_30"] = df.groupby(['store','item'])['sales'].shift(30)

df['rolling_mean_7'] = df.groupby(['store','item'])['sales'].shift(1).rolling(7).mean()
df['rolling_std_7'] = df.groupby(['store','item'])['sales'].shift(1).rolling(7).std()

df = df.dropna()

train = df[df['date'] < '2017-01-01']
test = df[df['date'] >= '2017-01-01']

features = ['store','item','day_of_week','month','year','lag_1','lag_7','lag_30','rolling_mean_7','rolling_std_7']

X_train = train[features]
y_train = train['sales']

X_test = test[features]
y_test = test['sales']

model = XGBRegressor(
    n_estimatiors=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("MAPE:",round(mape*100,2), "%")

joblib.dump(model, "models/demand_model.pkl")
print("Model saved")