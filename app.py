from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('Medinc_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        house_age = float(request.form['houseAge'])
        ave_rooms = float(request.form['aveRooms'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # Prepare and scale input
        input_data = np.array([[ave_rooms, house_age, latitude, longitude]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_text = round(prediction,2)

        return  render_template('home.html', prediction_text=round(prediction_text*1000,3))

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
