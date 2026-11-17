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
        image = sample["image"]
        
        # Support both VQA mode and fallback captioning mode
        if self.is_vqa and "question" in sample and "answer" in sample:
            question = sample["question"]
            answer = sample["answer"]
        else:
            # Fallback to pure image captioning mode using Flickr captions
            cap_idx = random.randint(0, 4)
            question = "describe the image"
            answer = sample[f"caption_{cap_idx}"]

        if self.transform:
            image = self.transform(image)

        # Build sequence: [<START>] + [question tokens] + [<SEP>] + [answer tokens] + [<END>]
        tokens = [self.vocab.stoi["<START>"]]
        tokens += self.vocab.numericalize(question)
        
        # Safely handle <SEP> token if present in vocab, otherwise fallback to <PAD> or space delimiter behavior
        sep_idx = self.vocab.stoi.get("<SEP>", None)
        if sep_idx is not None:
            tokens.append(sep_idx)
        
        answer_start_idx = len(tokens)  # Track exact index where answer tokens begin for loss masking
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