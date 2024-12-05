import os
import json
import numpy as np
import torch
from torch.utils.data import  Dataset

class SpeechtoTextDataset(Dataset):
    def __init__(self, mfcc_dir, text_dir, vocab_path=None, target_mfcc_length = 1380, target_text_length = 20, debug = False):
        """
        Initialize the dataset with MFCC and text directories, build or load the vocabulary.

        Args:
            mfcc_dir (str): Directory containing MFCC feature files.
            text_dir (str): Directory containing corresponding text files.
            vocab_path (str): Path to save/load vocabulary as a JSON file.
            target_mfcc_length (int): Target length to pad/truncate MFCC features.
            target_text_length (int): Target length to pad/truncate tokenized text.
            debug (bool): If True, prints debug information during data loading.
        """
        
        self.mfcc_dir = mfcc_dir
        self.text_dir = text_dir
        self.mfcc_files = [f for f in os.listdir(self.mfcc_dir) if f.lower().endswith(".npy")]
        self.text_files = [f for f in os.listdir(self.text_dir) if f.lower().endswith(".txt")]
        vocab_path = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\vocab.json'
        self.vocab_path = vocab_path

        # Ensure MFCC and text file counts match
        assert len(self.mfcc_files) == len(self.text_files), "Mismatch between MFCC and Text files count."
        print("")
        print(f"Found {len(self.mfcc_files)} MFCC files and {len(self.text_files)} text files.")
        
        # Load or build vocabulary
        self.vocab = self.load_or_build_vocab()
        self.output_size = len(self.vocab)
        print("")
        print(f"Vocabulary built with {self.output_size} words.")

        self.target_mfcc_length = target_mfcc_length
        self.target_text_length = target_text_length
        self.debug = debug

    def __len__(self):
        return len(self.mfcc_files)

    # Extract the spoken text from a formatted text file.
    def extract_text_from_file(self, text:str):
        
        lines = text.split("\n")
        for line in lines:
            if line.startswith("Text: "):
                return line.replace("Text: ", "").strip()
        return ""

    # Load an existing vocabulary or build a new one if not found.
    def load_or_build_vocab(self):
        if os.path.exists(self.vocab_path):
            with open (self.vocab_path, 'r') as f:
                vocab = json.load(f)
                print(f"Loaded vocabulary from {self.vocab_path}.")
        else: 
            vocab = self.build_vocab()
            self.save_vocab(vocab)
        return vocab
    
    # Save the vocabulary to a JSON file.
    def save_vocab(self, vocab):
        with open(self.vocab_path, 'w') as f:
            json.dump(vocab, f)
            print(f"\nVocabulary saved to {self.vocab_path}.")  
 
    # Update the existing vocabulary with any new words.
    def update_vocab(self, vocab):
        with open(self.vocab_path, 'r') as f:
            existing_vocab = json.load(f)
        new_words = {word: index for word, index in vocab.items() if word not in existing_vocab}
        if new_words:
            existing_vocab.update(new_words)
            with open(self.vocab_path, 'w') as f:
                json.dump(existing_vocab, f)
            print("\nVocabulary updated with new words.")

    # Build a vocabulary from the text files.
    def build_vocab(self):
        word_count = {}
        for text_file in self.text_files:
            with open(os.path.join(self.text_dir,text_file), 'r') as f:
                text = f.read().strip()
                spoken_text = self.extract_text_from_file(text)
                words = spoken_text.lower().split()
                for word in words:
                    word_count[word] = word_count.get(word,0) + 1 
        vocab = {'<PAD>' : 0, '<UNK>' : 1 } # <PAD> for padding, <UNK> for unknown words
        index  = 2
        for word, count in sorted(word_count.items(), key=lambda item: item[1], reverse = True):
            vocab[word] = index
            index += 1

        return vocab

    # Convert a word to its corresponding index.
    def word_to_index(self, word):
        return self.vocab.get(word, self.vocab['<UNK>']) # Default to <UNK> if word not in vocab

    # Tokenize the input text and convert to indices.
    def tokenize_text(self, text:str):
        tokens = text.lower().split()
        token_indices = [self.word_to_index(token) for token in tokens]
        return token_indices

    # Load and return the MFCC and tokenized text for a given index.
    def __getitem__(self, idx):
        mfcc_file = os.path.join(self.mfcc_dir, self.mfcc_files[idx])
        mfcc : np.ndarray = np.load(mfcc_file)
        if self.debug:
            print(f"Loaded MFCC File {mfcc_file}; MFCC shape is {mfcc.shape}.")

        if mfcc.shape[0] == 2:
            mfcc_reshaped = np.mean(mfcc , axis=0)
        else:
            mfcc_reshaped = mfcc[0]

        mfcc_reshaped = np.transpose(mfcc_reshaped, (1,0)) # Adjust MFCC shape for consistency

        mfcc_padded = self.pad_mfcc(mfcc_reshaped)
        if self.debug:
            print(f"Padded MFCC Shape: {mfcc_padded.shape}")

        text_file = os.path.join(self.text_dir, self.text_files [idx])
        with open (text_file, 'r') as f:
            text = f.read().strip()
        if self.debug:
                print(f"Loaded text file: {text_file}, content: '{text}'")
        
        spoken_text = self.extract_text_from_file(text)

        tokenized_text = self.tokenize_text(spoken_text)
        if self.debug:
                print(f"Original text: '{spoken_text}'")
                print(f"Tokenized text: {tokenized_text}")

        tokenized_text_padded = self.pad_text(tokenized_text)
        if self.debug:
                print(f"Padded tokenized text: {tokenized_text_padded}")

        return torch.FloatTensor(mfcc_padded), torch.LongTensor(tokenized_text_padded)

    # Pad or truncate the MFCC array to the target length.
    def pad_mfcc (self, mfcc:np.ndarray):
        if (mfcc.shape[0] < self.target_mfcc_length):
            pad_width = self.target_mfcc_length - mfcc.shape[0]
            mfcc_padded = np.pad(mfcc, ((0,pad_width) , (0,0)), mode = 'constant')
        else: 
            mfcc_padded = mfcc[: self.target_mfcc_length, :]

        return mfcc_padded

    # Pad or truncate the tokenized text to the target length.
    def pad_text(self, tokenized_text):
        if len(tokenized_text) < self.target_text_length:
            tokenized_text_padded = tokenized_text + [self.vocab['<PAD>']] * (self.target_text_length - len(tokenized_text))
        else:
            tokenized_text_padded = tokenized_text[:self.target_text_length]

        return tokenized_text_padded

    # Decode a list of token indices back into words.
    def decode(self, token_indices):
        index_to_word = {index: word for word, index in self.vocab.items()}
        decoded_words = [index_to_word.get(index, '') for index in token_indices if index != 0]
        return ' '.join(decoded_words).strip()


if __name__ == "__main__":
    mfcc_dir = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MFCC_FILES'
    text_dir = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\SPLIT_TEXT'
    vocab_path = r'C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\vocab.json'
    
    dataset = SpeechtoTextDataset(mfcc_dir, text_dir, vocab_path, debug=True)
    print(f"Dataset size: {len(dataset)}")

    for idx in range(len(dataset)):
        mfcc, tokenized_text = dataset[idx]
        print({f"MFCC Shape: {mfcc.shape}"})