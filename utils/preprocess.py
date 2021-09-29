import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
import spacy
import os


def txt2df(data_path):
    queries = []
    with open(data_path, "r") as input_txt:
        for line in input_txt.readlines():
            cur_line = line.strip("\n").split("\t")
            queries.append(cur_line)

    df = pd.DataFrame(queries,
                      columns=["session_id", "type", "prod_id", "query", "role"])
    return df


def filter_query_by_punct(puncts, auto_queries, s):
    if s in auto_queries:
        return True
    elif len(set(s).intersection(set(puncts))) == 0:
        return False
    else:
        return True


def filter_queries(df, max_char_len=50):
    puncts = ["-", "☞", "★", "---", "<url>", "→"]
    auto_queries = ["用户发起转人工", "售后咨询组"]

    # remove <url> and .jpg queries
    df = df[~(df["query"].str.contains('|'.join([".jpg", "<url>"])))]

    tqdm.pandas()
    df["filtered"] = df["query"].progress_apply(lambda x: filter_query_by_punct(puncts, auto_queries, x))

    df_filtered = df[df["filtered"] == False]

    # filter by char len
    df_filtered.loc[:, "char_len"] = df_filtered["query"].progress_apply(lambda x: len(x))
    df_filtered_l50 = df_filtered[df_filtered["char_len"] <= max_char_len]

    return df_filtered_l50


class spacyTokenizer(object):
    def __init__(self):
        self.tokenizer = spacy.load('zh_core_web_lg')

    def get_tokens(self, s):

        sent = self.tokenizer(s)
        tokens = []
        for token in sent:
            tokens.append(token.text)
        return tokens



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, dest='data_path', default='./data/data_train.txt',
                        help="input txt path")
    parser.add_argument('--unit', type=str, dest='unit', default="char",
                        help="tokenization unit", choices=["char", "word"])
    parser.add_argument("--split", type=bool, dest='split', default=True,
                        help="if split into train/dev/test")
    args = parser.parse_args()


    df = txt2df(args.data_path)

    print("filter queries....")
    df_filtered = filter_queries(df, max_char_len=50)


    if args.unit == "word":
        print("tokenize queries to words...")
        tokenizer = spacyTokenizer()
        df_filtered["tokens"] = df_filtered["query"].progress_apply(lambda x: "|".join(tokenizer.get_tokens(x)))

    if args.split:
        print("split dataset into train/test/dev....")
        train, test = train_test_split(df_filtered, shuffle=True, test_size=0.2)
        train, dev = train_test_split(train, shuffle=True, test_size=0.2)

        data_folder = os.path.dirname(args.data_path)

        train.to_csv(os.path.join(data_folder, "train.csv"))
        dev.to_csv(os.path.join(data_folder, "dev.csv"))
        test.to_csv(os.path.join(data_folder, "test.csv"))
        print("dataset saved to %s" % data_folder)
    else:
        df_filtered.to_csv(args.data_path.replace(".txt", ".csv"))






