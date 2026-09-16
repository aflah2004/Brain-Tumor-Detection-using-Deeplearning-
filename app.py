import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load your trained model
MODEL_PATH = 'BrainTumor10Epochs.h5'
model = load_model(MODEL_PATH)

# Store patient data
patients_list = []


# --------- MODEL HELPERS ---------
def get_className(classNo):
    return "Brain Tumor Detected" if classNo == 1 else "No Evidence of Brain Tumor"

def getResult(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((64, 64))
    img = np.array(img) / 255.0
    input_img = np.expand_dims(img, axis=0)
    predictions = model.predict(input_img)
    classNo = int(predictions[0][0] > 0.5)
    return classNo


# --------- ESTIMATE PRESSURE FUNCTION ---------
def estimate_pressure(img_path):
    """Estimate intracranial pressure (ICP) based on image intensity (simulated)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mean_intensity = np.mean(img)

    # Simulated mapping (for demo)
    pressure = round(5 + (mean_intensity / 255) * 25 + random.uniform(-2, 2), 2)

    # Safe bounds
    pressure = max(5, min(pressure, 30))
    return pressure


# --------- MAIN ROUTES ---------
@app.route('/', methods=['GET', 'POST'])
def patient():
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['age'] = request.form.get('age')
        session['gender'] = request.form.get('gender')
        session['contact'] = request.form.get('contact')
        session['email'] = request.form.get('email')
        return redirect(url_for('predict'))
    return render_template('patient.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file uploaded!"

        f = request.files['file']
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        filename = secure_filename(f.filename)
        file_path = os.path.join(upload_dir, filename)
        f.save(file_path)

        # Predict tumor
        class_no = getResult(file_path)
        result = get_className(class_no)
        pressure = estimate_pressure(file_path)  # 🧠 Estimate ICP
        img_url = url_for('static', filename=f'uploads/{filename}')

        patient_data = {
            'name': session.get('name'),
            'age': session.get('age'),
            'gender': session.get('gender'),
            'contact': session.get('contact'),
            'email': session.get('email'),
            'result': result,
            'pressure': pressure,
            'image': f'uploads/{filename}'
        }
        patients_list.append(patient_data)

        return render_template('predict.html', result=result, pressure=pressure, img_url=img_url)
    return render_template('predict.html', result=None, pressure=None)


@app.route('/patients')
def patients():
    return render_template('patients.html', patients=patients_list)


@app.route('/delete/<int:index>')
def delete(index):
    if 0 <= index < len(patients_list):
        patients_list.pop(index)
    return redirect(url_for('patients'))


@app.route('/edit/<int:index>', methods=['GET', 'POST'])
def edit(index):
    if 0 <= index < len(patients_list):
        patient = patients_list[index]
        if request.method == 'POST':
            patient['name'] = request.form.get('name')
            patient['age'] = request.form.get('age')
            patient['gender'] = request.form.get('gender')
            patient['contact'] = request.form.get('contact')
            patient['email'] = request.form.get('email')
            return redirect(url_for('patients'))
        return render_template('edit_patient.html', patient=patient, index=index)
    return redirect(url_for('patients'))


# --------- AI PREVENTION CHAT ---------
@app.route('/prevention', methods=['GET', 'POST'])
def prevention():
    suggestions = []
    user_input = ""

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').lower()

        if any(word in user_input for word in ['stress', 'tired', 'sleep']):
            suggestions = [
                "🧘 Try deep breathing or light yoga daily.",
                "💤 Maintain a 7–8 hour sleep schedule.",
                "🌿 Listen to calming music to relax your mind."
            ]
        elif any(word in user_input for word in ['diet', 'food', 'eat']):
            suggestions = [
                "🥦 Add vegetables, fruits, and fish oil to your diet.",
                "🍫 Limit sugar and processed snacks.",
                "💧 Stay hydrated throughout the day."
            ]
        elif any(word in user_input for word in ['screen', 'mobile', 'phone']):
            suggestions = [
                "👁️ Follow the 20-20-20 rule to relax your eyes.",
                "📵 Reduce blue light exposure at night.",
                "🕶️ Use anti-glare lenses if you work on screens long hours."
            ]
        else:
            suggestions = [
                "🏃 Stay active with at least 30 minutes of exercise daily.",
                "🩺 Get regular checkups to monitor brain health.",
                "🚭 Avoid smoking and alcohol for a healthier lifestyle."
            ]

    return render_template('prevention.html', suggestions=suggestions, user_input=user_input)


# --------- MAIN RUN ---------
if __name__ == '__main__':
    app.run(debug=True, port=5003)
