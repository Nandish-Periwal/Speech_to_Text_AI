from pathlib import Path
import json
import torch
import torchaudio
from SCRIPTS.TRAIN.MODEL import SpeechToTextModel
 
# Set torch print options for better output readability
torch.set_printoptions(precision=4, sci_mode=False)

# Define file paths for the audio, model, and vocab
audio_file_name = "test_audio_3.wav"
audio_file_directory = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\TEST_DATA")
audio_file_path = audio_file_directory / audio_file_name
model_path = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MODEL\SPEECH_TO_TEXT_MODEL_TRAINED.pth")
vocab_path = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\vocab.json")
DECIMAL_PLACES = 4

# Function to extract MFCC features from the audio file
def extract_mfcc_from_audio (audio_file_path, n_mfcc=26) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file_path) # Load the audio file
    mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 552, "n_mels": 26}
        )
    mfcc = mfcc_transform(waveform) # Apply MFCC transformation
    print("")
    print(f"Original MFCC Shape: {mfcc.shape}") 
    print(f"Original MFCC dimension: {mfcc.ndim}") 

    if mfcc.shape[0] == 2: # If stereo, average the two channels
            mfcc = mfcc.mean(dim = 0)

    mfcc = mfcc.permute(1, 0 )  # Rearrange the dimensions
    print("")
    print(f"Transformed MFCC Shape: {mfcc.shape}") # Print the transformed MFCC shape
    print(f"TRansformed MFCC dimension: {mfcc.ndim}") # Print the transformed MFCC dimension
    return mfcc
     
def load_model(model_path):
    model = SpeechToTextModel()  
    state_dict = torch.load(model_path, weights_only=True) 
    model.load_state_dict(state_dict) 
    model.eval() 
    return model

# Function to decode predicted tokens into words using a vocabulary
def decode_tokens(tokens: torch.Tensor, vocab_path: str, threshold=0.1):
    with open(vocab_path, 'r') as f:
        word_to_index = json.load(f) # Load vocab as a dictionary
        index_to_word = {v: k for k, v in word_to_index.items()}

    # Convert tokens to words, filtering by probability threshold
    words = [
        index_to_word.get(int(token), "<UNK>") if prob >= threshold else "<UNK>"
        for token, prob in zip(tokens.flatten(), tokens.flatten())
        if int(token) != 0
    ] 
    return " ".join(words)

def run_inference(audio_file_path, model_path, vocab_path):
    mfcc = extract_mfcc_from_audio(audio_file_path)
    mfcc = mfcc.unsqueeze(0) 

    model = load_model(model_path)

    with torch.inference_mode():
        output = model(mfcc)
        output = output.squeeze(0)

        probabilities = torch.softmax(output, dim=1)
        max_probs, predicted_tokens = torch.max(probabilities, dim=1)

        threshold = 0.0
        predicted_tokens[max_probs < threshold] = 1

        transcript = decode_tokens(predicted_tokens, vocab_path)
    print(f"\nTranscript: {transcript}")
    return transcript

run_inference(audio_file_path, model_path, vocab_path)

