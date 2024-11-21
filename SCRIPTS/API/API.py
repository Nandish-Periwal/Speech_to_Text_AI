from flask import Flask, request, jsonify
from pathlib import Path
import torch
import torchaudio
import json
import sys
from flask_cors import CORS


sys.path.append(str(Path(__file__).resolve().parents[1]))  # This adds the parent directory to sys.path

from SCRIPTS.TRAIN.MODEL import SpeechToTextModel

app = Flask(__name__)
CORS(app)

# Paths to your files
model_path = Path("D:/Coding/practice/AI/MODEL/SPEECH_TO_TEXT_MODEL_TRAINED.pth")
vocab_path = Path("D:/Coding/practice/AI/VOCAB/vocab.json")

# Load the model at startup
model = SpeechToTextModel()
try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Ensure temp directory exists
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

# Function to extract MFCC features
def extract_mfcc_from_audio (audio_file_path, n_mfcc=26) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file_path)
    mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 552, "n_mels": 26}
        )
    mfcc = mfcc_transform(waveform)
    print("")
    print(f"Original MFCC Shape: {mfcc.shape}")
    print(f"Original MFCC dimension: {mfcc.ndim}")

    if mfcc.shape[0] == 2:
            mfcc = mfcc.mean(dim = 0)

    # if mfcc.shape[1] < target_length:
    #       padding = target_length - mfcc.shape[1]
    #       mfcc = torch.nn.functional.pad(mfcc, (0, padding))
    # elif mfcc.shape[1] > target_length:
    #       mfcc = mfcc[:, :target_length]

    mfcc = mfcc.permute(1, 0 )
    print("")
    print(f"Transformed MFCC Shape: {mfcc.shape}")
    print(f"TRansformed MFCC dimension: {mfcc.ndim}")

    return mfcc

# Function to decode tokens
def decode_tokens(tokens: torch.Tensor, vocab_path: Path):
    with open(vocab_path, 'r') as f:
        word_to_index = json.load(f)
    index_to_word = {v: k for k, v in word_to_index.items()}
    return " ".join(index_to_word.get(int(token), "<UNK>") for token in tokens if int(token) != 0)

# Inference endpoint
@app.route("/transcribe/", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    audio_path = temp_dir / file.filename
    file.save(audio_path)
    print(f"File saved to {audio_path}")

    # Extract MFCC and run inference
    mfcc = extract_mfcc_from_audio(audio_path)
    if mfcc is None:
        return jsonify({"error": "Error processing audio file"}), 500

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    with torch.no_grad():
        output = model(mfcc)
        probabilities = torch.softmax(output.squeeze(0), dim=1)
        max_probs, predicted_tokens = torch.max(probabilities, dim=1)
        predicted_tokens[max_probs < 0.1] = 1  # Apply threshold for low-confidence predictions

    transcript = decode_tokens(predicted_tokens, vocab_path)
    return jsonify({"transcript": transcript})

if __name__ == "_main_":
    app.run(host="127.0.0.1", port=5000, debug=True)