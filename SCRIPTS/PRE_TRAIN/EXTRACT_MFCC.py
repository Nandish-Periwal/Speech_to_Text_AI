import os
import torch
import torchaudio
import numpy as np

# transforming waveform into mfcc features(tensors)
def extract_mfcc_from_audio(audio_file_path, n_mfcc=26):
    """Extract MFCC features from an audio file using torchaudio."""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Print the sample rate and waveform shape
    # print(f"Sample Rate: {sample_rate}")
    # print(f"Waveform shape: {waveform.shape}")
    # print(f"Waveform values: \n{waveform.numpy()}")  # Optional, but can be very large for long audio files
    
    # Define the MFCC transformation
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 1024, "hop_length": 552, "n_mels": 26}
    )
    
    # Apply the MFCC transformation
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.squeeze(0).numpy()  # Convert to NumPy array
    

    print(f"MFCC shape for {audio_file_path}: {mfcc.shape}")
  
    
    return mfcc

def process_audio_segments(processed_audio_dir, features_dir, n_mfcc=26):
    """Process all audio segments and save their MFCC features."""
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    audio_files = [f for f in os.listdir(processed_audio_dir) if f.lower().endswith(".wav")]

    for audio_file in audio_files:
        audio_file_path = os.path.join(processed_audio_dir, audio_file)
        print(f"Extracting MFCC from: {audio_file_path}")
        
# Extract MFCC using torchaudio
        mfcc_features = extract_mfcc_from_audio(audio_file_path, n_mfcc=n_mfcc)

# saving mfcc feature extracted from segmented audios        
# Save the MFCC features as a .npy file
        mfcc_file_name = audio_file.replace('.wav', '.npy')
        mfcc_file_path = os.path.join(features_dir, mfcc_file_name)
        np.save(mfcc_file_path, mfcc_features)
        
        print(f"MFCC saved to: {mfcc_file_path}")

processed_audio_dir = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\PROCESSED_AUDIO'
features_dir = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MFCC_FILES'

if not os.path.exists(features_dir):
    os.makedirs(features_dir)

process_audio_segments(processed_audio_dir, features_dir)