import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
class HuggingFaceDataset(IterableDataset):
    def __init__(self):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset to load.
            tokenizer_name (str): Name of the tokenizer to use.
            split (str): Which split of the dataset to use (e.g., 'train', 'test', 'validation').
            max_length (int): Maximum length of the tokenized sequences.
            buffer_size (int): Buffer size for shuffling the streamed dataset.
        """
        train_dataset = load_dataset("./data/test_dataset/", split="train")
        train_dataset = train_dataset.shuffle()
        self.dataset = train_dataset.select_columns("token_ids")

    def __iter__(self):
        buffer = []
        for sample in self.dataset.shuffle():
            inputs = sample["token_ids"]
            yield {
                'input_ids': inputs
            }

def get_dataset():
    dataset = HuggingFaceDataset()
    return dataset
