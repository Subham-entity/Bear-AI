import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MultimodalDataset(Dataset):
    def __init__(self, dataset, vocab, transform=None, is_vqa=True):
        self.dataset = dataset
        self.vocab = vocab
        self.transform = transform
        self.is_vqa = is_vqa

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")  # Ensure 3 channels[cite: 11]

        if self.is_vqa:
            question = sample["question"]
            if "answer" in sample and isinstance(sample["answer"], str):
                answer = sample["answer"]
            elif "multiple_choice_answer" in sample:
                answer = sample["multiple_choice_answer"]
            elif "answers" in sample:
                ans_list = sample["answers"]
                answer = ans_list[0]["answer"] if isinstance(ans_list[0], dict) else ans_list[0]
            else:
                answer = ""
        else:
            # Captioning mode: use the exact default prompt expected by app_2.py / inference_2.py[cite: 9, 10]
            question = "describe the image"
            cap_idx = random.randint(0, 4)
            answer = sample[f"caption_{cap_idx}"]

        if self.transform:
            image = self.transform(image)

        # Build sequence: [<START>] + [question/prompt] + [<SEP>] + [answer/caption] + [<END>][cite: 11]
        tokens = [self.vocab.stoi["<START>"]]
        tokens += self.vocab.numericalize(question)
        tokens.append(self.vocab.stoi["<SEP>"])

        answer_start_idx = len(tokens)
        tokens += self.vocab.numericalize(answer)
        tokens.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(tokens, dtype=torch.long), answer_start_idx

# Alias preserved for backwards compatibility with existing imports
FlickrDataset = MultimodalDataset


class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        sequences = [item[1] for item in batch]
        answer_start_idxs = torch.tensor([item[2] for item in batch], dtype=torch.long)

        # Pad sequences to the max length in the batch[cite: 11]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)

        return images, padded_sequences, answer_start_idxs