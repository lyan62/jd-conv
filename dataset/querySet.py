from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils.utils import pad_seq, build_dict, creat_output_dir
import os
import pickle


class querySet(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, index_col=0)
        self.dict_path =  "./output/dict.pkl"
        self.dict = self.build_dict()

    def __getitem__(self, index):
        cur_row = self.df.iloc[index]
        query = [self.dict["source"].get(c, 1) for c in cur_row["query"]]
        label = self.dict["target"].get(str(cur_row["role"]))
        text = cur_row["query"]
        return query, label, text

    def __len__(self):
        return len(self.df)

    def build_dict(self):
        if not os.path.exists(self.dict_path):
            dictionary = build_dict(self.df)
            creat_output_dir("output")
            pickle.dump(dictionary, open(self.dict_path, "wb"))
            return dictionary
        else:
            return self.load_dict(self.dict_path)

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
    train_set = querySet("./data/train.csv")
    from torch.utils.data import Dataset, DataLoader
    data_loader = DataLoader(train_set, batch_size=3, shuffle=True, collate_fn=my_collate_fn,
                             num_workers=4, pin_memory=True, sampler=None)

    for batched_data, lengths in data_loader:
        print(batched_data, lengths)
        break

    # dictionary = build_dict(train_set.df)
    # creat_output_dir("output")
    #
    # import pickle
    # import os
    # pickle.dump(dictionary, open(os.path.join("./output/", "dict.pkl"), "wb"))