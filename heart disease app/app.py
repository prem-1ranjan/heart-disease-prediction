from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Raw input
    Age = float(request.form["Age"])
    RestingBP = float(request.form["RestingBP"])
    Cholesterol = float(request.form["Cholesterol"])
    FastingBS = float(request.form["FastingBS"])
    MaxHR = float(request.form["MaxHR"])
    Oldpeak = float(request.form["Oldpeak"])
    Sex_M = float(request.form["Sex_M"])
    ChestPainType = request.form["ChestPainType"]
    RestingECG = request.form["RestingECG"]
    ExerciseAngina = request.form["ExerciseAngina"]
    ST_Slope = request.form["ST_Slope"]

    # One hot encoding manually
    data = {
        "Age": Age, "RestingBP": RestingBP,
        "Cholesterol": Cholesterol, "FastingBS": FastingBS,
        "MaxHR": MaxHR, "Oldpeak": Oldpeak, "Sex_M": Sex_M,
        "ChestPainType_ATA": 1 if ChestPainType == "ATA" else 0,
        "ChestPainType_NAP": 1 if ChestPainType == "NAP" else 0,
        "ChestPainType_TA":  1 if ChestPainType == "TA"  else 0,
        "RestingECG_Normal": 1 if RestingECG == "Normal" else 0,
        "RestingECG_ST":     1 if RestingECG == "ST"     else 0,
        "ExerciseAngina_Y":  1 if ExerciseAngina == "Y"  else 0,
        "ST_Slope_Flat":     1 if ST_Slope == "Flat"     else 0,
        "ST_Slope_Up":       1 if ST_Slope == "Up"       else 0,
    }

    df = pd.DataFrame([data])[columns]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]

    return render_template("index.html", prediction=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)