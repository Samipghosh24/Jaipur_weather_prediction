from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib 

app = Flask(__name__)
weather = pd.read_csv(
    "Weather_jaipur.csv", 
    index_col='Date', 
    usecols=['Date', 'Max_Temperature', 'Avg_Temperature', 'Min_Temperature', 'Max_Humidity', 'Avg_Humidity', 'Min_Humidity', 'Max_Wind_Speed', 'Avg_Wind_Speed', 'Min_Wind_Speed']
)
weather.index = pd.to_datetime(weather.index)
weather["Target_Temp"] = weather.shift(-1)["Max_Temperature"]
weather["Target_Humid"] = weather.shift(-1)["Max_Humidity"]
weather["Target_Speed"] = weather.shift(-1)["Max_Wind_Speed"]
weather = weather.ffill()

predictors = weather.columns[~weather.columns.isin(["Target_Temp", "Target_Humid", "Target_Speed", "Date"])]

rr_temp = Ridge(alpha=0.1)
rr_temp.fit(weather[predictors], weather["Target_Temp"])

rr_humid = Ridge(alpha=0.1)
rr_humid.fit(weather[predictors], weather["Target_Humid"])

rr_speed = Ridge(alpha=0.1)
rr_speed.fit(weather[predictors], weather["Target_Speed"])

# X = weather[predictors]
# y_temp = weather["Target_Temp"]
# y_humid=weather["Target_Humid"]
# y_speed=weather["Target_Speed"]
# pred_temp = rr_temp.predict(X)
# pred_humid = rr_humid.predict(X)
# pred_speed = rr_speed.predict(X)

# print("MAE (Temperature):", mean_absolute_error(y_temp, pred_temp))
# print("MAE (Humidity):", mean_absolute_error(y_humid, pred_humid))
# print("MAE (speed):", mean_absolute_error(y_speed, pred_speed))

joblib.dump(rr_temp, 'rr_temp.pkl')
joblib.dump(rr_humid, 'rr_humid.pkl')
joblib.dump(rr_speed, 'rr_speed.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    user_input = pd.DataFrame([data])
    rr_temp = joblib.load('rr_temp.pkl')
    rr_humid = joblib.load('rr_humid.pkl')
    rr_speed = joblib.load('rr_speed.pkl')
    temp_pred = rr_temp.predict(user_input)[0]
    humid_pred = rr_humid.predict(user_input)[0]
    speed_pred = rr_speed.predict(user_input)[0]
    return jsonify({
        'predicted_temperature': temp_pred,
        'predicted_humidity': humid_pred,
        'predicted_wind_speed': speed_pred
    })

if __name__ == '__main__':
    app.run(debug=True)


