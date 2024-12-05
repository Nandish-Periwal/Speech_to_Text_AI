import torch
import torch.nn as nn
import torch.nn.functional as F
from SCRIPTS.TRAIN.DATASET import SpeechtoTextDataset


class SpeechToTextModel(nn.Module):
    def __init__(self, input_size=26, hidden_size = 128, intermediate_size = 128, output_size = 136, num_layers = 2):
        """
        Initialize the Speech-to-Text model with GRU and fully connected layers.
        Args:
            input_size (int): Number of input features (MFCC features).
            hidden_size (int): Number of hidden units in GRU layers.
            intermediate_size (int): Hidden units in intermediate fully connected layer.
            output_size (int): Vocabulary size or number of output classes.
            num_layers (int): Number of GRU layers.
        """
        super(SpeechToTextModel, self).__init__()
        # GRU layer for sequential data processing
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout = 0.2)
        # Fully connected layers for transforming GRU outputs
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU() # Activation function for non-linearity
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # GRU output, ignore hidden state
        out = self.fc2(self.relu(self.fc1(gru_out))) # Apply FC layers with ReLU activation
        return out
    
    def test_model(self, sample_input, dataset):
        self.eval() 
        with torch.no_grad(): 
            output = self.forward(sample_input)
            print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_length, output_size)
            probabilities = F.softmax(output, dim=-1) # Convert logits to probabilities
            predicted_indices = torch.argmax(probabilities, dim=-1) # Get the most likely class indices

            print(f"Predicted indices shape: {predicted_indices.shape}")


