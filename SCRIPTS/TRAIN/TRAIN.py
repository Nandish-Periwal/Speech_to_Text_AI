import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from SCRIPTS.TRAIN.DATASET import SpeechtoTextDataset
from SCRIPTS.TRAIN.MODEL import SpeechToTextModel
from pathlib import Path
import torchaudio

# Hyperparameters
NUM_EPOCHS = 161
BATCH_SIZE = 24
MAX_LENGTH = 1380
LEARNING_RATE = 0.01

# Directory paths for MFCC files, text files, and model saving location
mfcc_dir = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MFCC_FILES"
text_dir = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\SPLIT_TEXT"
model_dir = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MODEL") 
if not os.path.exists(model_dir):
    os.makedirs(model_dir) 

# Custom collate function for padding MFCC and text sequences in the batch.
def collate_fn(batch):
    mfccs, texts = zip(*batch)  

    # Pad MFCC sequences to a fixed length
    padded_mfccs = nn.utils.rnn.pad_sequence(
        [torch.tensor(mfcc)[:MAX_LENGTH] for mfcc in mfccs], batch_first=True
    )
    
    # Pad text sequences (tokenized text) to the longest sequence in the batch
    max_text_len = max([text.size(0) for text in texts])  
    padded_texts = torch.stack([
        F.pad(text, (0, max_text_len - text.size(0)), value=0)  # Pad with zeros (index 0 for padding token)
        for text in texts
    ])
    
    return padded_mfccs, padded_texts 

# Compute accuracy by comparing predicted and target labels.
def compute_accuracy(outputs, targets): 
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

# Convert prediction tokens to text based on the vocabulary.
def token_to_text(pred_tokens):
    vocab = dataset.build_vocab()
    for word, index in vocab():
        if index == pred_tokens:
            print (word)

# Train the Speech-to-Text model for a number of epochs.
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
            target_lengths = [len(text) for text in texts] 
            
            # Mask to ignore time steps beyond target lengths
            max_target_len = max(target_lengths)
            mask = torch.zeros_like(outputs, dtype=torch.bool)

            for i, length in enumerate(target_lengths):
                mask[i, :length, :] = True  

            # Apply the mask
            outputs_masked = outputs[mask].view(-1, outputs.shape[-1])  # Shape: (batch_size * seq_length, output_size)
            texts_flat = torch.cat([text for text in texts], dim=0)  # Flatten the targets)
            
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
                    print(f"outputs_masked shape: {outputs_masked.shape}")  
                    print(f"texts_flat shape: {texts_flat.shape}") 
                    print(f'Epoch [{epoch}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}')

    MODEL_NAME = "SPEECH_TO_TEXT_MODEL_TRAINED.pth"
    MODEL_SAVE_PATH = model_dir / MODEL_NAME

    print(f"Saving Model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

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

    audio_batch, text_batch = next(iter(dataloader))
    audio_single , text_single = audio_batch[1], text_batch[1]
    
    model.eval()
    with torch.inference_mode():
        pred = model(audio_single)
        print(pred.ndim) # 2
        print(pred.shape) # 1380, 136
        pred_tokens = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual text:\n{text_single}")

    decoded_text = dataset.decode(pred_tokens.tolist())
    print(f"Decoded text: '{decoded_text}'")
