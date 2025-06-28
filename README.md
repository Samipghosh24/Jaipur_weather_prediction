# Jaipur_weather_prediction

This project has 3 parts Dataset,python code and webpage

# Dataset
The data set contains weather data over Jaipur,Rajasthan from the period 01/08/2014 to 31/07/2024 with total 3654 data points

The dataset contains 16 different attributes including dates

# Python Code
For prediction we are using the following attributes: Max_Temperature, Avg_Temperature, Min_Temperature, Max_Humidity, Avg_Humidity, Min_Humidity, Max_Wind_Speed, Avg_Wind_Speed, Min_Wind_Speed

Ridge regression model is used for prediction, it ia a simple model but is capable of providing good prediction

The MSE of the model for temperature,humidity and wind speed is 1.31,6.29 and 5.81 respectively

The predicted output is stored in .pkl file that is accessed on the web page for displaying the prediction

# Webpage
It is a simple webpage that takes max,min temperature, min,max humidity and min,max wind speed prom the user for predictions

It displays predicted temperature,humidity and wind speed for the coresponding data
