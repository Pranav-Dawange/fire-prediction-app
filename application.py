import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    print(">>> [DEBUG] GET / route hit")
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            print(">>> [DEBUG] POST /predictdata route hit")

            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_data_scaled = standard_scaler.transform(input_data)
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', results=f"The predicted value is {result[0]:.2f}")
        except Exception as e:
            print(">>> [ERROR] Exception during prediction:", e)
            return render_template('home.html', results="Error during prediction.")
    else:
        print(">>> [DEBUG] GET /predictdata route hit")
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
