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
        
        # Flickr8k provides 5 captions; let's grab the first one for simplicity, 
        # or you can randomly sample one during training for data augmentation!
        caption = sample["caption"][0]

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
        # batch is a list of tuples: [(image1, caption1), (image2, caption2), ...]
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        # Stack images into a single tensor: (Batch_Size, 3, 224, 224)
        images = torch.stack(images, dim=0)

        # Pad captions to the longest sequence in this batch
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions