import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FlickrDataset(Dataset):
    def __init__(self, dataset, vocab, transform=None, is_vqa=False):
        self.dataset = dataset
        self.vocab = vocab
        self.transform = transform
        self.is_vqa = is_vqa

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")  # Ensure 3 channels (some VQA images are grayscale)
        question = sample["question"]
        
        # Handle datasets with either a single 'answer' string or a list of 'answers'
        if "answer" in sample and isinstance(sample["answer"], str):
            answer = sample["answer"]
        elif "multiple_choice_answer" in sample:
            answer = sample["multiple_choice_answer"]
        elif "answers" in sample:
            # Pick the most frequent or first answer from the list
            ans_list = sample["answers"]
            answer = ans_list[0]["answer"] if isinstance(ans_list[0], dict) else ans_list[0]
        else:
            answer = ""

        if self.transform:
            image = self.transform(image)

        # Build sequence: [<START>] + [question] + [<SEP>] + [answer] + [<END>]
        tokens = [self.vocab.stoi["<START>"]]
        tokens += self.vocab.numericalize(question)
        tokens.append(self.vocab.stoi["<SEP>"])
        
        answer_start_idx = len(tokens)
        tokens += self.vocab.numericalize(answer)
        tokens.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(tokens), answer_start_idx


class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        sequences = [item[1] for item in batch]
        answer_start_idxs = torch.tensor([item[2] for item in batch], dtype=torch.long)

        # Pad sequences to the max length in the batch
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)

        return images, padded_sequences, answer_start_idxs