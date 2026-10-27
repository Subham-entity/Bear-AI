import collections
import torch

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Initialize dictionaries for word-to-index and index-to-word
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Basic lowercase and whitespace tokenizer
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                # Add word to vocab if it meets the frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]