# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model (KNN for Heart Disease Prediction)
filename = 'heart-disease-prediction-model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract and convert form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Prepare data for prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])
        
        # Make prediction
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    # Ensure compatibility with Flask ≥ 2.2
    app.run(debug=True, host='0.0.0.0', port=5000)
