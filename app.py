import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('model.pkl', 'rb'))  # Trained Polynomial Regression model
scaler = pickle.load(open('scaler.pkl', 'rb'))   # MinMaxScaler used for Weekly_Sales

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        holiday_flag = int(request.form['Holiday_Flag'])
        unemployment = float(request.form['Unemployment'])
        holiday_impact = float(request.form['holiday_impact'])
        holiday_impact_sq = holiday_impact ** 2

        # Create input DataFrame as used during training
        input_df = pd.DataFrame([{
            'Holiday_Flag': holiday_flag,
            'Unemployment': unemployment,
            'holiday_impact': holiday_impact,
            'holiday_impact^2': holiday_impact_sq
        }])

        # Predict scaled Weekly_Sales
        scaled_prediction = regmodel.predict(input_df)[0]

        # Inverse transform to get actual Weekly_Sales
        final_prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]

        return render_template('form.html', prediction_text=f"📊 Predicted Weekly Sales: ${final_prediction:,.2f}")
    
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

# ✅ This block should be at the end, and NOT indented
if __name__ == '__main__':
    app.run(debug=True)
