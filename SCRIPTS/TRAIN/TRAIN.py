import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from DATASET import SpeechtoTextDataset
from MODEL import SpeechToTextModel
from pathlib import Path
import torchaudio
# import matplotlib.pyplot as plt

NUM_EPOCHS = 161
BATCH_SIZE = 24
MAX_LENGTH = 1380
LEARNING_RATE = 0.01

mfcc_dir = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MFCC_FILES"
text_dir = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\SPLIT_TEXT"
model_dir = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MODEL/" 
if not os.path.exists(model_dir):
    os.makedirs(model_dir) 

def collate_fn(batch):
    mfccs, texts = zip(*batch)  # Unpack the batch of MFCCs and text transcriptions

    # Pad MFCC sequences to the maximum length defined
    padded_mfccs = nn.utils.rnn.pad_sequence(
        [torch.tensor(mfcc)[:MAX_LENGTH] for mfcc in mfccs], batch_first=True
    )
    
    # Pad text sequences (tokenized text) to the longest sequence in the batch
    max_text_len = max([text.size(0) for text in texts])  # Find the maximum text length
    padded_texts = torch.stack([
        F.pad(text, (0, max_text_len - text.size(0)), value=0)  # Pad with zeros (index 0 for padding token)
        for text in texts
    ])
    
    return padded_mfccs, padded_texts  # Return the padded MFCC and text tensors

def compute_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

def token_to_text(pred_tokens):
    vocab = dataset.build_vocab()
    for word, index in vocab():
        if index == pred_tokens:
            print (word)

def train(model, dataloader, criterion, optimizer, num_epochs):
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
    
        for batch in dataloader:
            mfccs, texts = batch
            
            # Forward pass
            outputs = model(mfccs)  # Shape: (batch_size, seq_length, output_size)

            # Get the actual sequence length of the targets
            target_lengths = [len(text) for text in texts]  # The actual lengths of the target texts
            
            # Create a mask for the outputs to ignore time steps beyond the length of the target text
            max_target_len = max(target_lengths)
            mask = torch.zeros_like(outputs, dtype=torch.bool)

            for i, length in enumerate(target_lengths):
                mask[i, :length, :] = True  

            # Apply the mask
            outputs_masked = outputs[mask].view(-1, outputs.shape[-1])  # Shape: (batch_size * seq_length, output_size)
            texts_flat = torch.cat([text for text in texts], dim=0)  # Flatten the targets
            
            # # Debug prints to check shapes
            # print(f"outputs_masked shape: {outputs_masked.shape}")  # Should be (N, output_size)
            # print(f"texts_flat shape: {texts_flat.shape}")  # Should be (N,)
            
            # Ensure the shapes of outputs and targets match
            assert outputs_masked.shape[0] == texts_flat.shape[0], \
                f"Mismatch in shape: outputs ({outputs_masked.shape[0]}), targets ({texts_flat.shape[0]})"
            
            # Compute loss
            loss = criterion(outputs_masked, texts_flat)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() 
            accuracy = compute_accuracy(outputs_masked, texts_flat)
            epoch_accuracy += accuracy

            train_losses.append(epoch_loss / len(dataloader))
            train_accuracies.append(epoch_accuracy / len(dataloader))


            model.eval()
            with torch.inference_mode():
                if (epoch % 20 == 0):
                    print(f"outputs_masked shape: {outputs_masked.shape}")  # Should be (N, output_size)
                    print(f"texts_flat shape: {texts_flat.shape}")  # Should be (N,)
                    print(f'Epoch [{epoch}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}')

    MODEL_NAME = "SPEECH_TO_TEXT_MODEL_TRAINED.pth"
    MODEL_SAVE_PATH = r"D:\Coding\practice\MODEL\SPEECH_TO_TEXT_MODEL_TRAINED.pth"

    # print(f"Saving Model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def extract_mfcc_from_audio (audio_file_path, n_mfcc=26) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file_path)
    mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 552, "n_mels": 26}
        )
    mfcc = mfcc_transform(waveform)

    if mfcc.shape[0] == 2:
            mfcc = mfcc.mean(dim = 0)

    mfcc = mfcc.permute(1, 0 )

    return mfcc

if __name__ == "__main__":
#     # Initialize the dataset and DataLoader with the specified directories
    dataset = SpeechtoTextDataset(mfcc_dir=mfcc_dir, text_dir=text_dir) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

#     # Initialize the model, loss function, and optimizer
    model = SpeechToTextModel() 
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # Start training the model
    train(model, dataloader, criterion, optimizer, NUM_EPOCHS)

#     # Save the trained model
    torch.save(model.state_dict(), f'{model_dir} speech_to_text_model.pth')  # Save the model in the MODELS directory
    # print(f"Model saved to {model_dir} speech_to_text_model.pth")

    audio_batch, text_batch = next(iter(dataloader))
    audio_single , text_single = audio_batch[1], text_batch[1]
    
    model.eval()
    with torch.inference_mode():
        pred = model(audio_single)
        print(pred.ndim) # 2
        print(pred.shape) # 1380, 136
        pred_tokens = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        
    # print(f"Output logits:\n{pred}\n")
    # print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Actual text:\n{text_single}")

    decoded_text = dataset.decode(pred_tokens.tolist())
    print(f"Decoded text: '{decoded_text}'")



    # audio_file_path = r"C:\Users\MyLaptopKart\Desktop\ai_practice\office test.WAV"
    # audio_test = extract_mfcc_from_audio(audio_file_path=audio_file_path, n_mfcc=26)

    # print(f"Audio test shape: {audio_test.shape}")
    # model.eval()
    # with torch.inference_mode():
    #     predict = model(audio_test)
    #     print(predict.ndim)
    #     predict_tokens = torch.argmax(torch.softmax(predict, dim=1), dim=1)
        
    # # # print(f"Output logits:\n{predict}\n")
    # # # print(f"Output prediction probabilities:\n{torch.softmax(predict, dim=1)}\n")
    # print(f"Output prediction label:\n{predict_tokens}\n")
    # # # print(f"Actual text:\n{text_single}")

    # subtitle = dataset.decode(predict_tokens.tolist())
    # print(f"Decoded text: '{subtitle}'")