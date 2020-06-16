import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
root = ''
split = 0.8


class WikiSQL(Dataset):

    def __init__(self, text, sql, transform=None):
        """
        Args:
            text (string): File location of text to be converted to sql
            sql (string): File location of corresponding sql queries
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        with open(text, 'rb') as f:
            self.text = pickle.load(f, encoding='bytes')
        with open(sql, 'rb') as file_b:
            self.sql = pickle.load(file_b, encoding='bytes')

        self.max_len_text = 0
        self.max_len_sql = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # sample = {'text': torch.FloatTensor(self.text[idx]), 'sql': torch.FloatTensor(self.sql[idx])}
        text = torch.FloatTensor(self.text[idx])
        sql = torch.FloatTensor(self.sql[idx])

        self.max_len_text = max(self.max_len_text, len(text))
        self.max_len_sql = max(self.max_len_sql, len(sql))

        # if self.transform:
        #     sample = self.transform(sample)
        return [text, sql]

    def collate(self, batch):
        text = []
        sql = []
        for n, x in enumerate(batch):
            text.append(torch.cat((x[0], torch.zeros(self.max_len_text - x[0].size(0))), dim=0))
            sql.append(torch.cat((x[1], torch.zeros(self.max_len_sql - x[1].size(0))), dim=0))
        self.max_len_text, self.max_len_sql = 0, 0

        return torch.stack(text), torch.stack(sql)


# def collate(batch):
#     batch = np.array(batch)
#     print('collate', batch[0])
#     # max_len_text = 0
#     # max_len_sql = 0
#     # for n, text, sql in enumerate(batch):
#     #
#     #     max_len_text = max(max_len_text, len(text))
#     #     max_len_sql = max(max_len_sql, len(sql))
#
#     return batch


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         text, sql = sample['text'], sample['sql']
#         # k_nearest_pixels = np.array(k_nearest_pixels)
#         return {'noisy': torch.FloatTensor(noisy),
#                 'ground_truth': torch.FloatTensor(ground_truth)}

if __name__ == '__main__':

    questions_path = 'data/questions/'
    sql_queries_path = 'data/sql_queries/'
    word_idx_mappings_path = 'data/word_idx_mappings/'
    wiki_sql_path = 'data/WikiSQL_files/'
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    transformed_dataset = WikiSQL(text=os.path.join(questions_path, 'train_questions_tokenized.pkl'),
                                  sql=os.path.join(sql_queries_path, 'train_sql_tokenized.pkl'),
                                  transform=compose)

    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)

    for x in dataloader:
        print(x['text'])
        print(x['sql'])
        print(torch.unique(x['train_text']))
