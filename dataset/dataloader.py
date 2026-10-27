import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FlickrDataset(Dataset):
    def __init__(self, dataset, vocab, transform=None):
        self.dataset = dataset
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        
        # Randomly select one of the 5 available captions for data augmentation
        cap_idx = random.randint(0, 4)
        caption = sample[f"caption_{cap_idx}"]

        if self.transform:
            image = self.transform(image)

        # Numericalize caption and wrap with <START> and <END> tokens
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numericalized_caption)


class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions