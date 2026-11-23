# -------------------- Windows PyTorch DLL fix --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU on Windows

# -------------------- Imports --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from flask import Flask, request, render_template
import numpy as np
import wfdb

print("Torch version:", torch.__version__)

# -------------------- Config --------------------
MODEL_PATH = "best_cnn_arrhythmia(1).pt"
INPUT_LENGTH = 360
NUM_CLASSES = 5
CLASS_ORDER = ['N', 'S', 'V', 'F', 'Q']
CHANNEL_INDEX = 0

# -------------------- CNN1D Model Definition --------------------
class CNN1D(nn.Module):
    def __init__(self, input_length=INPUT_LENGTH, num_classes=NUM_CLASSES, dropout_p=0.5):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(dropout_p)

        self.conv3 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(dropout_p)

        self.flattened_size = 128 * (input_length // 8)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.drop_fc1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = self.fc2(x)
        return x

# -------------------- Load trained model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully!")

# -------------------- Flask app --------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -------------------- Helper functions --------------------
def preprocess_beat(beat):
    beat = np.array(beat, dtype=np.float32)
    std = np.std(beat)
    if std == 0 or np.isnan(std):
        beat = beat - np.mean(beat)
    else:
        beat = (beat - np.mean(beat)) / (std + 1e-8)
    return beat

def detect_rpeaks(signal_channel, fs):
    try:
        from wfdb import processing
        return processing.gqrs_detect(sig=signal_channel, fs=fs)
    except Exception:
        return np.array([], dtype=int)

def aami_map_symbol_to_index(sym):
    symbol_map = {
        'N':'N','L':'N','R':'N','e':'N','j':'N',
        'A':'S','a':'S','J':'S','S':'S',
        'V':'V','E':'V',
        'F':'F',
        '/':'Q','f':'Q','Q':'Q','?':'Q','~':'Q',
        '|':'Q','!':'Q','[':'Q',']':'Q','x':'Q',
        '(':'Q',')':'Q','p':'Q'
    }
    mapped = symbol_map.get(sym, None)
    if mapped is None:
        return None
    return CLASS_ORDER.index(mapped)

def read_record_and_annotation(tmpdir, basename):
    path = os.path.join(tmpdir, basename)
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal
    fs = int(rec.fs)
    r_peaks = symbols = None
    try:
        ann = wfdb.rdann(path, 'atr')
        r_peaks = ann.sample
        symbols = ann.symbol
    except Exception:
        pass
    return sig, fs, r_peaks, symbols

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return "No files uploaded", 400

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = {}
        for f in uploaded_files:
            fname = f.filename
            if not fname:
                continue
            fpath = os.path.join(tmpdir, fname)
            f.save(fpath)
            name, ext = os.path.splitext(fname)
            saved.setdefault(name, []).append(ext.lower())

        chosen = None
        for base, exts in saved.items():
            if ".dat" in exts and ".hea" in exts:
                chosen = base
                break
        if chosen is None:
            return "Upload matching .dat and .hea files", 400

        # Read record
        try:
            sig, fs, r_peaks, symbols = read_record_and_annotation(tmpdir, chosen)
        except Exception as e:
            return str(e), 500

        # Select channel
        channel = sig[:, CHANNEL_INDEX] if sig.ndim==2 else sig

        # Detect R-peaks if missing
        if r_peaks is None or len(r_peaks)==0:
            r_peaks = detect_rpeaks(channel, fs)
        if r_peaks is None or len(r_peaks)==0:
            return "No R-peaks detected", 500

        # Extract beats
        half_w = INPUT_LENGTH // 2
        beats = []
        beat_symbol_list = []
        for idx, r in enumerate(r_peaks):
            start = int(r) - half_w
            end = int(r) + half_w
            if start < 0 or end > len(channel):
                continue
            beat = preprocess_beat(channel[start:end])
            beats.append(beat)
            beat_symbol_list.append(symbols[idx] if symbols is not None and idx < len(symbols) else None)

        if len(beats)==0:
            return "No valid beats extracted", 500

        # Predict in batch
        X_tensor = torch.from_numpy(np.stack(beats, axis=0)).float().unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        # Prepare results for display
        results = []
        for i in range(len(beats)):
            p = probs[i]
            pred_idx = int(np.argmax(p))
            conf = float(p[pred_idx])
            results.append({
                "index": i+1,
                "pred_label": CLASS_ORDER[pred_idx],
                "confidence": f"{conf*100:.2f}%",
                "orig_symbol": beat_symbol_list[i]
            })

        # Compute BPM
        diffs = np.diff(r_peaks).astype(np.float64)
        bpm = 60.0 / (np.mean(diffs)/fs) if len(diffs) > 0 else 0.0

        # Render results page
        return render_template("results.html", results=results, bpm=f"{bpm:.2f}", num_beats=len(beats))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
