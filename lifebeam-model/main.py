from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('ModelBeta.h5')
model2 = tf.keras.models.load_model("hKenaModelBeta.h5")

# Load the trained scaler
scaler_filename = 'scalerBeta.pkl'
scaler_filename2 = 'hKenaScalerBeta.pkl'
scaler = pickle.load(open(scaler_filename, 'rb'))
scaler2 = pickle.load(open(scaler_filename2, 'rb'))

# Prediction route
@app.route('/predict/calorie-offset', methods=['POST'])
def predictCalorieOffset():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            json_data = request.json

            # Extract input values from JSON data
            gender = int(json_data['gender'])
            age = float(json_data['age'])
            height = float(json_data['height'])
            weight = float(json_data['weight'])
            exercise = float(json_data['exercise'])
            calories = float(json_data['calories'])
            day = float(json_data['day'])
            bmi = float(json_data['bmi'])
            cno = float(json_data['cno'])
            clu = float(json_data['clu'])
            bmr = float(json_data['bmr'])
            cintake = float(json_data['cintake'])
            cburned = float(json_data['cburned'])
            eused = float(json_data['eused'])

            # Create a numpy array from the input values
            input_data = np.array([[gender, age, height, weight, exercise, calories, day, bmi, cno, clu, bmr, cintake, cburned, eused]])

            # Preprocess the input data using the loaded scaler
            input_scaled = scaler.transform(input_data)

            # Make predictions using the loaded model
            prediction = model.predict(input_scaled).flatten()[0]

            # Convert prediction to Python float
            prediction = float(prediction)

            return jsonify({'calorieOffset': prediction})

        else:
            return jsonify({'error': 'Invalid content type. Expected application/json.'})

@app.route('/predict/day-left', methods=['POST'])
def predictDayLeft():
    try:
        # Parse JSON data from the request
        json_data = request.json

        # Extract input values from JSON data
        bmi = float(json_data['bmi'])
        cno = float(json_data['cno'])
        clu = float(json_data['clu'])
        weight = float(json_data['weight'])
        height = float(json_data['height'])
        sisa_calories = float(json_data['calorieOffset'])

        if sisa_calories > 0:
            if bmi >= 25:
                return jsonify({'dayLeft': 0})
            
            if bmi < 18.5:
                hKena = (((18.5 * height * height) - weight) * 7700) / sisa_calories
                return jsonify({'dayleft': hKena})

            # Continue with the prediction logic
            input_data = pd.DataFrame({'BMI': [bmi], 'CNO': [cno], 'CLU': [clu], 'sisaCalories': [sisa_calories]})
            input_scaled = scaler2.transform(input_data)
            prediction = model2.predict(input_scaled)
            result = float(prediction[0][0])
        else:
            if bmi < 18.5:
                return jsonify({'dayLeft': 0})
            
            if bmi >= 25:
                hKena = ((weight - (24.9 * height * height)) * (-7700)) / sisa_calories
                return jsonify({'dayLeft': hKena})
            
            # Continue with the prediction logic
            input_data = pd.DataFrame({'BMI': [bmi], 'CNO': [cno], 'CLU': [clu], 'sisaCalories': [sisa_calories]})
            input_scaled = scaler2.transform(input_data)
            prediction = model2.predict(input_scaled)
            result = float(prediction[0][0])

        return jsonify({'dayLeft': result})

    except Exception as e:
        return jsonify({'error': str(e), 'result': 0})

if __name__ == "__main__":
    app.run(debug=True)