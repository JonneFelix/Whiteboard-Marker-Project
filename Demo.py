import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from flask import Flask, request, render_template_string, send_file
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from itertools import product
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Download nltk words corpus
nltk.download('words')

# Set matplotlib backend to Agg
plt.switch_backend('Agg')

app = Flask(__name__)

# Load the model
model_path = 'Modell.h5'
model = load_model(model_path)

# Dictionary to convert numbers to letters
zahlen_zu_buchstaben = {i: chr(ord('A') + i) for i in range(26)}

# Load the German words dictionary
german_words = set(w.lower() for w in nltk.corpus.words.words() if w.isalpha())

# Default parameters
default_params = {
    'threshold': 0.3,
    'window_size': 90,
    'start_offset': 0.6,
    'end_offset': 0.6
}

def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def quaternion_to_rotation_matrix(row):
    quat = [row['i'], row['j'], row['k'], row['r']]
    r = R.from_quat(quat)
    return r.as_matrix()

def remove_gravity(data):
    cleaned_accel = np.zeros((len(data), 3))
    for index, row in data.iterrows():
        rotation_matrix = quaternion_to_rotation_matrix(row)
        accel = np.array([row['x'], row['y'], row['z']])
        cleaned_accel[index] = rotation_matrix @ accel
    data['cleaned_x'] = cleaned_accel[:, 0]
    data['cleaned_y'] = cleaned_accel[:, 1]
    data['cleaned_z'] = cleaned_accel[:, 2]
    return data

def apply_savgol_filter(data, window_length=11, polyorder=2):
    data['cleaned_x'] = savgol_filter(data['cleaned_x'], window_length, polyorder)
    data['cleaned_y'] = savgol_filter(data['cleaned_y'], window_length, polyorder)
    data['cleaned_z'] = savgol_filter(data['cleaned_z'], window_length, polyorder)
    return data

def detect_pauses(data, threshold, window_size, start_offset, end_offset):
    data['cleaned_z_std'] = data['cleaned_z'].rolling(window=window_size).std()
    movement_start = (data['cleaned_z_std'] > threshold)
    movement_end = (data['cleaned_z_std'] < threshold)

    pause_times = []
    post_pause_times = []
    is_moving = False

    for i in range(1, len(movement_start)):
        if movement_start.iloc[i] and not is_moving:
            start_time = data['timestamp'].iloc[i] - pd.Timedelta(seconds=start_offset)
            is_moving = True
        elif movement_end.iloc[i] and is_moving:
            end_time = data['timestamp'].iloc[i] - pd.Timedelta(seconds=end_offset)
            pause_times.append(start_time)
            post_pause_times.append(end_time)
            is_moving = False

    return pause_times, post_pause_times

def segment_data(data, pause_times, post_pause_times):
    segments = []
    for start, end in zip(pause_times, post_pause_times):
        segment = data[(data['timestamp'] >= start) & (data['timestamp'] < end)]
        if not segment.empty:
            segments.append(segment)
    return segments

def preprocess_and_predict(model, segment):
    new_data = segment.drop(["timestamp", "x", "y", "z", "la", "r", "i", "j", "k", "qa", "cleaned_z_std"], axis=1)
    new_data = new_data.dropna()
    new_data = new_data.astype(np.float32)
    new_data_tensor = tf.convert_to_tensor(new_data.values)
    new_data_tensor = tf.pad(new_data_tensor, paddings=[[0, 400 - tf.shape(new_data_tensor)[0]], [0, 0]])
    new_data_tensor = tf.expand_dims(new_data_tensor, axis=0)
    prediction = model.predict(new_data_tensor)
    predicted_prob = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]
    top_3_indices = prediction[0].argsort()[-3:][::-1]
    top_3_probs = prediction[0][top_3_indices]
    top_3_classes = [zahlen_zu_buchstaben[i] for i in top_3_indices]
    return zahlen_zu_buchstaben[predicted_class], predicted_prob, list(zip(top_3_classes, top_3_probs)), prediction[0]

def generate_combinations(probabilities):
    letters = [list(map(lambda x: x[0], prob)) for prob in probabilities]
    return list(product(*letters))

def calculate_probability(word, probabilities):
    prob = 1.0
    for i, letter in enumerate(word):
        letter_prob = dict(probabilities[i]).get(letter, 0)
        prob *= letter_prob
    return prob

def find_best_word(probabilities, dictionary):
    best_word = None
    best_prob = 0
    combinations = generate_combinations(probabilities)
    for word in combinations:
        word_str = ''.join(word).lower()
        if word_str in dictionary:
            prob = calculate_probability(word, probabilities)
            if prob > best_prob:
                best_prob = prob
                best_word = word_str
    return best_word, best_prob

def plot_segment(segment, prediction):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(segment['timestamp'], segment['cleaned_x'], label='Cleaned X')
    ax.plot(segment['timestamp'], segment['cleaned_y'], label='Cleaned Y')
    ax.plot(segment['timestamp'], segment['cleaned_z'], label='Cleaned Z')
    ax.set_title(f'Segment Visualization - Predicted Class: {prediction[0]} with Probability: {prediction[1]:.2f}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Acceleration')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def plot_complete_data(data, pause_times, post_pause_times):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data['timestamp'], data['cleaned_x'], label='Cleaned X')
    ax.plot(data['timestamp'], data['cleaned_y'], label='Cleaned Y')
    ax.plot(data['timestamp'], data['cleaned_z'], label='Cleaned Z')
    for start, end in zip(pause_times, post_pause_times):
        ax.axvspan(start, end, color='red', alpha=0.3)
    ax.set_title('Complete Data with Segments')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Acceleration')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def create_pdf(predicted_word):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.drawString(100, 750, predicted_word)
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

@app.route('/', methods=['GET', 'POST'])
def index():
    global default_params

    if request.method == 'POST':
        try:
            # Retrieve the file from the form
            file = request.files['file']

            print(f"Received file: {file.filename}")
            print(f"File content type: {file.content_type}")

            if file.filename == '':
                return "<h1>No file selected</h1>"

            try:
                data = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file, encoding='ISO-8859-1')
                except UnicodeDecodeError:
                    data = pd.read_csv(file, encoding='latin1')

            print(data.head())

            data['timestamp'] = pd.to_datetime(data['timestamp'])

            threshold = float(request.form.get('threshold', default_params['threshold']))
            start_offset = float(request.form.get('start_offset', default_params['start_offset']))
            end_offset = float(request.form.get('end_offset', default_params['end_offset']))

            data = remove_gravity(data)
            data = apply_savgol_filter(data)
            data[['cleaned_x', 'cleaned_y', 'cleaned_z']] = normalize_data(
                data[['cleaned_x', 'cleaned_y', 'cleaned_z']]
            )

            pause_times, post_pause_times = detect_pauses(
                data,
                threshold=threshold,
                window_size=default_params['window_size'],
                start_offset=start_offset,
                end_offset=end_offset
            )

            segments = segment_data(data, pause_times, post_pause_times)

            all_predictions = []
            all_top_3 = []
            segment_images = []

            for segment in segments:
                predicted_class, predicted_prob, top_3, full_probs = preprocess_and_predict(model, segment)
                all_predictions.append((predicted_class, predicted_prob))
                all_top_3.append(top_3)
                image_base64 = plot_segment(segment, (predicted_class, predicted_prob))
                segment_images.append(image_base64)

            best_word, best_prob = find_best_word(all_top_3, german_words)
            complete_data_image = plot_complete_data(data, pause_times, post_pause_times)

            results = '''
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f4f4f4;
                        color: #333;
                    }}
                    .container {{
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                        background-color: #fff;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    h1, h2, h3 {{
                        color: #555;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                    }}
                    .segment {{
                        margin-bottom: 20px;
                    }}
                    .segment img {{
                        border: 1px solid #ddd;
                        padding: 10px;
                        background-color: #fafafa;
                    }}
                    .segment h3 {{
                        margin: 0 0 10px;
                        padding: 10px;
                        background-color: #eee;
                        border-left: 5px solid #ddd;
                    }}
                    .form-group {{
                        margin-bottom: 15px;
                    }}
                    .form-group label {{
                        display: block;
                        margin-bottom: 5px;
                    }}
                    .form-group input {{
                        width: 100%;
                        padding: 8px;
                        box-sizing: border-box;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    .form-group input[type="submit"] {{
                        background-color: #007bff;
                        color: white;
                        border: none;
                        cursor: pointer;
                    }}
                    .form-group input[type="submit"]:hover {{
                        background-color: #0056b3;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Prediction Result</h1>
                    <h2>Complete Data Visualization</h2>
                    <img src="data:image/png;base64,{}" alt="Complete Data Visualization">
                    <h2>Segment Predictions</h2>
            '''.format(complete_data_image)

            for i, (predicted_class, predicted_prob) in enumerate(all_predictions):
                results += '''
                    <div class="segment">
                        <h3>Segment {}: Predicted letter: {} with probability {:.4f}</h3>
                        <img src="data:image/png;base64,{}" alt="Segment Visualization">
                        <ul>
                '''.format(i + 1, predicted_class, predicted_prob, segment_images[i])
                for letter, prob in all_top_3[i]:
                    results += "<li>Letter: {} - Probability: {:.4f}</li>".format(letter, prob)
                results += "</ul></div>"

            results += "<h1 style='font-size: 2em;'>Best word: '{}' with probability {:.4f}</h1>".format(best_word,
                                                                                                         best_prob)
            if best_word:
                results += '''
                <form method="post" action="/download_pdf">
                    <input type="hidden" name="predicted_word" value="{}">
                    <input type="submit" value="Download PDF">
                </form>
                '''.format(best_word)

            results += '''
                </div>
                <div class="container">
                    <h1>Upload CSV File</h1>
                    <form method="post" enctype="multipart/form-data">
                        <div class="form-group">
                            <label for="file">Upload CSV File:</label>
                            <input type="file" name="file" id="file" required>
                        </div>
                        <div class="form-group">
                            <label for="threshold">Threshold:</label>
                            <input type="text" name="threshold" id="threshold" placeholder="Threshold" value="{0}">
                        </div>
                        <div class="form-group">
                            <label for="start_offset">Start Offset:</label>
                            <input type="text" name="start_offset" id="start_offset" placeholder="Start Offset" value="{1}">
                        </div>
                        <div class="form-group">
                            <label for="end_offset">End Offset:</label>
                            <input type="text" name="end_offset" id="end_offset" placeholder="End Offset" value="{2}">
                        </div>
                        <div class="form-group">
                            <input type="submit" value="Submit">
                        </div>
                    </form>
                </div>
            </body>
            </html>
            '''.format(threshold, start_offset, end_offset)

            return render_template_string(results)

        except Exception as e:
            print(f"Error occurred: {str(e)}")  # Print the error to the console for debugging
            return f"<h1>Error processing the file</h1><p>{str(e)}</p>"

    return '''
    <html>
    <head>
        <title>Upload CSV File</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1, h2 {
                color: #555;
            }
            .form-group {
                margin-bottom: 15px;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
            }
            .form-group input {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .form-group input[type="submit"] {
                background-color: #007bff;
                color: white;
                border: none;
                cursor: pointer;
            }
            .form-group input[type="submit"]:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload CSV File</h1>
            <form method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="file">Upload CSV File:</label>
                    <input type="file" name="file" id="file" required>
                </div>
                <div class="form-group">
                    <label for="threshold">Threshold:</label>
                    <input type="text" name="threshold" id="threshold" placeholder="Threshold" value="0.3">
                </div>
                <div class="form-group">
                    <label for="start_offset">Start Offset:</label>
                    <input type="text" name="start_offset" id="start_offset" placeholder="Start Offset" value="0.6">
                </div>
                <div class="form-group">
                    <label for="end_offset">End Offset:</label>
                    <input type="text" name="end_offset" id="end_offset" placeholder="End Offset" value="0.6">
                </div>
                <div class="form-group">
                    <input type="submit" value="Submit">
                </div>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    predicted_word = request.form.get('predicted_word', '')
    pdf_buffer = create_pdf(predicted_word)
    return send_file(pdf_buffer, as_attachment=True, download_name="predicted_word.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
