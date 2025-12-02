from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
import joblib
import json

app = Flask(__name__)

# Load Part-1 Model (Binary COPD Classification)
model_binary = load_model('Part-1/saved_model_binary/copd_binary_model.keras')
with open('Part-1/saved_model_binary/class_names_binary.json', 'r') as f:
    class_names_binary = json.load(f)
label_encoder_binary = joblib.load('Part-1/saved_model_binary/label_encoder_binary.pkl')

# Load Part-2 Model (COPD Staging)
class COPDModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(COPDModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model_staging_columns = joblib.load('Part-2/model_artifacts/model_columns.pkl')
model_staging_input_size = len(model_staging_columns)
model_staging_classes = np.load('Part-2/model_artifacts/classes.npy', allow_pickle=True)
model_staging_num_classes = len(model_staging_classes)

model_staging = COPDModel(model_staging_input_size, model_staging_num_classes)
model_staging.load_state_dict(torch.load('Part-2/model_artifacts/copd_model.pth', weights_only=False))
model_staging.eval()

scaler_staging = joblib.load('Part-2/model_artifacts/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/copd', methods=['GET', 'POST'])
def copd_analysis():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            input_data = pd.DataFrame([form_data])
            
            # Preprocess input data
            input_data = input_data.reindex(columns=model_staging_columns, fill_value=0)
            input_data = scaler_staging.transform(input_data)
            
            input_tensor = torch.FloatTensor(input_data)
            
            with torch.no_grad():
                outputs = model_staging(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
            
            stage = model_staging_classes[predicted.item()]
            prob_values = (probabilities[0].numpy() * 100).tolist()
            prob_labels = model_staging_classes.tolist()
            
            return render_template('result.html', 
                                 prediction=stage, 
                                 prob_values=prob_values,
                                 prob_labels=prob_labels,
                                 comorbidity_suggestion=None)
        except Exception as e:
            return render_template('copd.html', error=str(e), model_columns=model_staging_columns)
            
    return render_template('copd.html', model_columns=model_staging_columns)

@app.route('/lung_disease')
def lung_disease_detection():
    return render_template('lung_disease.html')

@app.route('/analyze_copd', methods=['POST'])
def analyze_copd():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio_file']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        import tempfile
        import os
        import librosa
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Load audio file with kaiser_fast resampling (same as training)
            audio, sr = librosa.load(tmp_path, sr=22050, res_type='kaiser_fast')
            
            # Pad or truncate to 3 seconds (same as training)
            target_length = 22050 * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract MFCC features and transpose
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T
            
            # Reshape for model (batch_size, time_steps, features)
            mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            
            # Predict (binary classification with sigmoid)
            prediction_prob = model_binary.predict(mfcc, verbose=0)[0][0]
            predicted_class = 1 if prediction_prob > 0.5 else 0
            confidence = float(prediction_prob if predicted_class == 1 else 1 - prediction_prob)
            
            diagnosis = label_encoder_binary.inverse_transform([predicted_class])[0]
            
            return jsonify({
                'diagnosis': diagnosis,
                'confidence': confidence
            })
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)