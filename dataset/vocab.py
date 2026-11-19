import collections
import re


class Vocabulary:
    def __init__(self, freq_threshold=3):
        # Initialize dictionaries for word-to-index and index-to-word
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>", 4: "<SEP>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "<SEP>": 4}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Lowercase and separate standalone punctuation marks while preserving contractions (e.g., dog's)
        cleaned = re.sub(r"([?.!,;:\"/()\[\]{}])", r" \1 ", text.lower())
        return cleaned.split()

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 5

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                # Add word to vocab if it meets the frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def build_vqa_vocabulary(self, question_list, answer_list):
        """
        Builds vocabulary from both question and answer lists for VQA training.
        """
        combined_sentences = list(question_list) + list(answer_list)
        self.build_vocabulary(combined_sentences)

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]