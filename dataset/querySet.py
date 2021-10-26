from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils.utils import pad_seq, creat_output_dir
import os
import pickle
from collections import Counter
from tqdm import tqdm


class querySet(Dataset):
    def __init__(self, data_path, query_col_name, target_col_name):
        self.df = pd.read_csv(data_path, index_col=0)
        self.query_col_name = query_col_name
        self.target_col_name = target_col_name
        self.dict_path =  "./output/dict.pkl"
        self.dict = self.build_dict()


    def __getitem__(self, index):
        cur_row = self.df.iloc[index]
        text = cur_row[self.query_col_name]
        label = self.dict["target"].get(str(cur_row[self.target_col_name]))
        query = [self.dict["source"].get(c, 1) for c in text]
        return query, label, text

    def __len__(self):
        return len(self.df)

    def build_dict(self):
        if not os.path.exists(self.dict_path):
            dictionary = self._build_dict()
            creat_output_dir("output")
            pickle.dump(dictionary, open(self.dict_path, "wb"))
            return dictionary
        else:
            return self.load_dict(self.dict_path)

    def _build_dict(self):
        print("building dictionary....")
        source_freqs = Counter()
        label_freqs = Counter()
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            for c in row[self.query_col_name]:
                source_freqs[str(c)] += 1
            label_freqs[str(row[self.target_col_name])] += 1

        s_word2index = {v: i + 2 for i, (v, c) in enumerate(list(source_freqs.items()))}
        s_word2index['<PAD>'] = 0
        s_word2index['<UNK>'] = 1

        t_word2index = {v: i for i, (v, c) in enumerate(list(label_freqs.items()))}
        data_dict = {'source': s_word2index,
                     'target': t_word2index}

        return data_dict

    @staticmethod
    def load_dict(dict_path):
        return pickle.load(open(dict_path, "rb"))



def my_collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)  # sort sentences in decreasing length

    input_seqs, labels, texts = zip(*batch)
    lengths = [len(s) for s in input_seqs]
    input_padded = np.asarray([pad_seq(s, max(lengths)) for s in input_seqs]).astype(float)
    target = np.asarray([labels])

    batched_data = {'input': input_padded,
                    'target': target,
                    'text': texts}

    return batched_data, lengths



if __name__ == "__main__":
    train_set = querySet("./data/crosswoz_chat_intent.csv",
                         query_col_name="msg",
                         target_col_name="intent")
    from torch.utils.data import Dataset, DataLoader
    data_loader = DataLoader(train_set, batch_size=3, shuffle=True, collate_fn=my_collate_fn,
                             num_workers=4, pin_memory=True, sampler=None)

    for batched_data, lengths in data_loader:
        print(batched_data, lengths)
        break

