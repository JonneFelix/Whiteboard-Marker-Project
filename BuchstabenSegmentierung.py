# Teile dieses Codes wurden mithilfen von ChatGPT erstellt oder bearbeitet

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

# Daten laden
data = pd.read_csv("A-30_2.csv")
# Zeitstempel in Datetime-Objekte konvertieren
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Funktion zur Normalisierung der Daten
def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

# Funktion zur Umwandlung der Quaternionen in Rotationsmatrizen
def quaternion_to_rotation_matrix(row):
    quat = [row['i'], row['j'], row['k'], row['r']]
    r = R.from_quat(quat)
    return r.as_matrix()

# Funktion zum Entfernen der Erdbeschleunigung
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

# Funktion zur Anwendung des Savitzky-Golay-Filters zur Glättung der Daten
def apply_savgol_filter(data, window_length=11, polyorder=2):
    data['cleaned_x'] = savgol_filter(data['cleaned_x'], window_length, polyorder)
    data['cleaned_y'] = savgol_filter(data['cleaned_y'], window_length, polyorder)
    data['cleaned_z'] = savgol_filter(data['cleaned_z'], window_length, polyorder)
    return data

# Funktion zur Erkennung von Pausen
def detect_pauses(data, threshold=0.005, window_size=100, start_offset=0, end_offset=0):
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

# Funktion zum Segmentieren der Daten basierend auf Pausen
def segment_data(data, pause_times, post_pause_times):
    segments = []
    for start, end in zip(pause_times, post_pause_times):
        segment = data[(data['timestamp'] >= start) & (data['timestamp'] < end)]
        if not segment.empty:
            segments.append(segment)
    return segments

# Funktion zum Speichern der Segmente in einem angegebenen Ordner
def save_segments(segments, labels, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, segment in enumerate(segments):
        filename = f'segment_{i}_label_{labels[i]}.csv'
        filepath = os.path.join(folder_path, filename)
        segment["Target"] = labels[i]
        segment.to_csv(filepath, index=False)

# Funktion zur Visualisierung der Daten
def plot_sensor_data(data, pause_times, post_pause_times, subset=True, normalize=True):
    if subset:
        total_time = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
        start_time = data['timestamp'].iloc[0] + total_time / 2 - pd.Timedelta(seconds=15)
        end_time = data['timestamp'].iloc[0] + total_time / 2 + pd.Timedelta(seconds=15)
        subset_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
    else:
        subset_data = data

    subset_data = remove_gravity(subset_data)
    subset_data = apply_savgol_filter(subset_data)

    if normalize:
        subset_data[['cleaned_x', 'cleaned_y', 'cleaned_z']] = normalize_data(
            subset_data[['cleaned_x', 'cleaned_y', 'cleaned_z']])

    plt.figure(figsize=(12, 12))

    for i, axis in enumerate(['X', 'Y', 'Z'], start=1):
        plt.subplot(3, 1, i)
        plt.plot(subset_data['timestamp'], subset_data[f'cleaned_{axis.lower()}'], label=f'Cleaned Acc{axis}')
        plt.title(f'Bereinigte Beschleunigung {axis}')
        plt.xlabel('Zeit')
        plt.ylabel('Normalisierte Beschleunigung' if normalize else 'Bereinigte Beschleunigung')
        plt.legend()
        for j, (pause, post_pause) in enumerate(zip(pause_times, post_pause_times)):
            plt.axvline(pause, color='red', linestyle='--', lw=1)
            plt.axvline(post_pause, color='blue', linestyle='--', lw=1)
            plt.axvspan(pause, post_pause, color='grey', alpha=0.3)
            plt.text(pause, plt.ylim()[0], str(j), color='red', fontsize=12, verticalalignment='bottom')

    plt.tight_layout()
    plt.show()

# Hauptfunktion
def main(data, folder_path, threshold=0.005, window_size=100, labels=None, start_offset=0, end_offset=0, save_csv=False):
    data = remove_gravity(data)
    data = apply_savgol_filter(data)
    data[['cleaned_x', 'cleaned_y', 'cleaned_z']] = normalize_data(data[['cleaned_x', 'cleaned_y', 'cleaned_z']])

    pause_times, post_pause_times = detect_pauses(data, threshold, window_size, start_offset, end_offset)
    segments = segment_data(data, pause_times, post_pause_times)

    if labels is None:
        labels = ['label'] * len(segments)

    if save_csv and folder_path:
        save_segments(segments, labels, folder_path)

    plot_sensor_data(data, pause_times, post_pause_times, subset=False, normalize=False)

folder_path = r"C:\Users\Hermes\PycharmProjects\MLunddasQSamPC\venv\csvTempStorage"

#labels = ["B","A","L","D","C","A","K","E","D","E","C","K","B","E","L","L","F","L","A","G","A","C","E","E","E","E"]
#labels = ["non","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","W","R","S","T","U","V","W","X","Y","Z"] * 100
#labels = ["D","A","S","I","S","T","E","I","N","T","E","S","T","none","Z","W","E","I"]
labels = ["H"] * 100
save_csv = False
main(data, folder_path, threshold=0.27, window_size=90, labels=labels, start_offset=0.6, end_offset=0.6, save_csv=save_csv) #0.2,0.5
