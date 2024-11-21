import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from SCRIPTS.TRAIN.DATASET import SpeechtoTextDataset

class SpeechToTextModel(nn.Module):
    def __init__(self, input_size=26, hidden_size = 128, intermediate_size = 128, output_size = 136, num_layers = 2):
        super(SpeechToTextModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout = 0.2)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc2(self.relu(self.fc1(gru_out)))
        return out
    
    def test_model(self, sample_input, dataset):
        self.eval()
        with torch.no_grad():
            output = self.forward(sample_input)
            # print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_length, output_size)
            probabilities = F.softmax(output, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)

            print(f"Predicted indices shape: {predicted_indices.shape}")

            for i, indices in enumerate(predicted_indices):
                decoded_sentence = dataset.decode(indices.tolist())
                print(f"Sample {i + 1} decoded output: {decoded_sentence}")

