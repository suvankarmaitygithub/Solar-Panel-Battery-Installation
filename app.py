from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        grid_reliability = float(request.form['grid_reliability'])
        avg_power_outage_duration = float(request.form['avg_power_outage_duration'])
        peak_power_requirement = float(request.form['peak_power_requirement'])
        off_peak_power_requirement = float(request.form['off_peak_power_requirement'])
        solar_generation_capacity = float(request.form['solar_generation_capacity'])
        excess_electricity = float(request.form['excess_electricity'])
        battery_capacity = float(request.form['battery_capacity'])
        battery_type = request.form['battery_type']
        
        battery_type_Lithium_ion = 1 if battery_type == 'Lithium-ion' else 0
        
        features = np.array([[grid_reliability, avg_power_outage_duration, peak_power_requirement, 
                              off_peak_power_requirement, solar_generation_capacity, 
                              excess_electricity, battery_capacity, battery_type_Lithium_ion]])
        
        features = scaler.transform(features)
        prediction = model.predict(features)
        
        return render_template('result.html', prediction='Yes' if prediction[0] == 1 else 'No')

if __name__ == '__main__':
    app.run(debug=True)
