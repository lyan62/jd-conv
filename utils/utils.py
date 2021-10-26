# helper functions
import random
import torch
from torch.autograd import Variable
import numpy as np
import time
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import pickle
import logging as log


use_cuda = torch.cuda.is_available()


def bucket_by_length(data):
    "bucket sentence by length. with step of 5"
    data.sort(key=lambda x: len(x[0]))
    len_data = [len(pair[0]) for pair in data]
    boundaries = [0] + [r for r in range(min(len_data), max(len_data), 5)] + [max(len_data) + 1]
    bins, _ = np.histogram(len_data, boundaries)
    indexes = np.cumsum(bins).tolist()
    buckets = [data[indexes[i]:indexes[i + 1]] for i in range(len(indexes) - 1)]
    return buckets


def pad_seq(seq, max_length):
    "pad sequence with zero"
    PAD_token = 0
    tokens = [c for c in seq]
    tokens += [PAD_token for i in range(max_length - len(tokens))]
    return tokens


def pad_target(seq, max_length):
    "pad sequence with zero"
    PAD_token = -1
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def shuffle_bucketed_data(data):
    for bucket in data:
        random.shuffle(bucket)
    random.shuffle(data)


def repackage_hidden(hidden, reset=False):
    if type(hidden) != tuple:
        if reset:
            return Variable(hidden.data.zero_()).cuda() if use_cuda else Variable(hidden.data.zero_())
        else:
            return Variable(hidden.data).cuda() if use_cuda else Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v, reset=reset) for v in hidden)


def eval_mask(output, target_lengths):  # mask by de_lemma_word real length
    mask = torch.zeros(output.size())
    for i, l in enumerate(target_lengths):
        mask[i, :l, :] = 1
    return Variable(mask.byte()).cuda() if use_cuda else Variable(mask.byte())  # masked_select takes variabel


def loss_mask(output, target_batches):  # mask by padded de_lemma_word length
    mask = torch.zeros(output.size())
    mask[:target_batches.size(0), :target_batches.size(1), :] = 1
    return Variable(mask.byte()).cuda() if use_cuda else Variable(mask.byte())  # return variable


def variableFromSentence(sent):
    v = Variable(torch.from_numpy(sent).long())  # MAXLEN * 1
    return v.cuda() if use_cuda else v


def print_progress(epoch, batch_count, val_loss):
    log.info('-' * 89)
    log.info('| epoch {:3d} | batch{: d} | valid loss {:5.4f}'
             .format(epoch, batch_count, val_loss))
    log.info('-' * 89)


def get_class_weights(data):
    class_count = {}
    for i in range(len(data)):
        target = data[i][1]
        if target in class_count.keys():
            class_count[target] += 1
        else:
            class_count[target] = 1
    weights = [1 / class_count[pair[1]] for pair in data]
    return weights


def get_decode_indexes(input_text):
    decode_indexes = []
    for i, t in enumerate(input_text):
        if '▁' in t:
            decode_indexes.append(i)
    return decode_indexes


# def decode_text(input_seqs, predictions, target_seqs, s_bpe2index, t_word2index, data_dir, bpe_model):
#     prediction_texts = []
#     target_texts = []
#     input_texts = []
#
#     sp = spm.SentencePieceProcessor()
#     bpe_model = os.path.join(data_dir, bpe_model)
#     sp.Load(bpe_model)
#
#     s_index2bpe = {i:v for v,i in s_bpe2index.items()}
#     t_index2word = {i:v for v,i in t_word2index.items()}
#
#     for (input, target, prediction) in zip(input_seqs, target_seqs, predictions):
#         input_text = [s_index2bpe[w] for w in input]
#         decoded_text = sp.DecodePieces(input_text)
#
#         target_seq = [target[0]]*len(decoded_text.split(" "))
#         decoded_indexes = get_decode_indexes(input_text)
#         prediction_seq = [prediction[i] for i in decoded_indexes]
#
#         target_text = [t_index2word[w] for w in target_seq]
#         prediction_text = [t_index2word[w] for w in prediction_seq]
#
#         target_texts.append(target_text)
#         prediction_texts.append(prediction_text)
#         input_texts.append(decoded_text.split(" "))
#     return input_texts, target_texts, prediction_texts

def save_text(input, target, prediction, outfile_dir):
    print("Saving text to file")
    filename = os.path.join(outfile_dir, "decoded_all.txt")
    with open(filename, "w+") as outfile:
        for (i, t, p) in zip(input, target, prediction):
            outfile.write("Input: " + i + "\n")
            for i in range(len(p)):
                if p[i] == t[i]:
                    p[i] = p[i] + "*"
            outfile.write("Target: " + " >> ".join(t) + "\n")
            outfile.write("Predictions:" + " >> ".join(p) + "\n")
    outfile.close()


def get_accuracy(pred, true):
    "get accuract of pred, true, each is a.txt sequence"
    right = 0
    cnt = 0
    for i in range(len(true)):
        for t in range(len(true[i])):
            cnt += 1
            if true[i][t] == pred[i][t]:
                right += 1.0
    return right / cnt


def get_acc_at_time(true, pred, out_dir):
    acc = []
    proportion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accfile = open(os.path.join(out_dir, "accuracy_at_timestep.txt"), "w+")
    for idx, p in enumerate(proportion):
        correct_cnt = 0
        for i in range(len(pred)):
            time_step = min(round(p * len(pred[i])), len(pred[i]) - 1)
            if pred[i][time_step] == true[i][time_step]:
                correct_cnt += 1
        current_acc = round(correct_cnt / len(pred), 2)
        acc.append(current_acc)
        out_text = "Sentence Revealed: " + str(p) + ", accuracy = " + str(acc[idx]) + "\n"
        accfile.write(out_text)
    accfile.close()


def get_predicted_all(input_ids, input_seq, pred_seq, conf_seq, true_seq, s_word2index, t_word2index, outfile_dir, file,
                      attn_seqs, p):
    s_index2word = {i: v for v, i in s_word2index.items()}
    t_index2word = {i: v for v, i in t_word2index.items()}
    input_all = []
    pred_all = []
    target_all = []
    conf_all = []
    label_all = []

    if len(attn_seqs) > 0:
        attn_all = []

    for idx in range(len(input_seq)):
        # recover text from source indexes
        input_list = [s_index2word[w] if w in s_index2word.keys() else '<UNK>' for w in input_seq[idx]]
        pred = t_index2word[pred_seq[idx]]
        conf = conf_seq[idx]
        target = t_index2word[true_seq[idx]]

        input_all.append(" ".join(input_list))
        pred_all.append(str(pred))
        conf_all.append(round(conf, 4))
        target_all.append(str(target))
        label_all.append(target == pred)

        if len(attn_seqs) > 0:
            attn_list = attn_seqs[idx]
            attn_all.append([round(a, 4) for a in attn_list])

    print("Writing attention samples to file...")
    if len(attn_seqs) > 0:
        df = pd.DataFrame({"preverbs": input_all,
                           "attn": attn_all,
                           "predictions": pred_all,
                           "confidences": conf_all,
                           "targets": target_all,
                           "labels": label_all,
                           "ids": input_ids,
                           "revealed": [p] * len(input_seq)},
                          columns=['ids', 'labels', 'preverbs', 'attn', 'targets', 'predictions', 'confidences',
                                   'revealed'])
    else:
        df = pd.DataFrame({"preverbs": input_all,
                           "predictions": pred_all,
                           "confidences": conf_all,
                           "targets": target_all,
                           "labels": label_all,
                           "ids": input_ids,
                           "revealed": [p] * len(input_seq)},
                          columns=['ids', 'labels', 'preverbs', 'targets', 'predictions', 'confidences', 'revealed'])
    sorted = df.sort_values(by='ids')
    sorted.to_csv(os.path.join(outfile_dir, file))


def get_predicted_multi(input_ids, input_seq, pred_seq, true_seq, s_word2index, t_word2index, outfile_dir, file,
                        attn_seqs, p):
    s_index2word = {i: v for v, i in s_word2index.items()}
    t_index2word = {i: v for v, i in t_word2index.items()}
    input_all = []
    pred_all = []
    target_all = []
    conf_all = []
    label_all = []

    if len(attn_seqs) > 0:
        attn_all = []

    for idx in range(len(input_seq)):
        # recover text from source indexes
        input_list = [s_index2word[w] if w in s_index2word.keys() else '<UNK>' for w in input_seq[idx]]
        pindices = [i for i, x in enumerate(pred_seq[idx]) if x == 1]
        tindices = [i for i, x in enumerate(true_seq[idx]) if x == 1]
        pred = [t_index2word[i] for i in pindices]
        target = [t_index2word[i] for i in tindices]

        input_all.append(" ".join(input_list))
        pred_all.append(" ".join(pred))
        target_all.append(" ".join(target))
        label_all.append(target == pred)

        if len(attn_seqs) > 0:
            attn_list = attn_seqs[idx]
            attn_all.append([round(a, 4) for a in attn_list])

    print("Writing attention samples to file...")
    if len(attn_seqs) > 0:
        df = pd.DataFrame({"preverbs": input_all,
                           "attn": attn_all,
                           "predictions": pred_all,
                           "targets": target_all,
                           "labels": label_all,
                           "ids": input_ids,
                           "revealed": [p] * len(input_seq)},
                          columns=['ids', 'labels', 'preverbs', 'attn', 'targets', 'predictions', 'revealed'])
    else:
        df = pd.DataFrame({"preverbs": input_all,
                           "predictions": pred_all,
                           "targets": target_all,
                           "labels": label_all,
                           "ids": input_ids,
                           "revealed": [p] * len(input_seq)},
                          columns=['ids', 'labels', 'preverbs', 'targets', 'predictions', 'revealed'])
    sorted = df.sort_values(by='ids')
    sorted.to_csv(os.path.join(outfile_dir, file))


def apply_inplace(df, field, func):
    return pd.concat([df.drop(field, axis=1), df[field].apply(func)], axis=1)


def get_predicted_syn(all_out, dictionary, outfile_dir, file, p):
    s_index2word = {i: v for v, i in dictionary['source'].items()}
    t_index2word = {i: v for v, i in dictionary['target'].items()}

    if 'attn_seqs' not in all_out.keys():
        all_out['attn_seqs'] = [None] * len(all_out['input_ids'])
    df = pd.DataFrame({
        "ids": all_out['input_ids'],
        "preverbs": all_out['input_seqs'],
        "predictions": all_out['predictions'],
        "confidences": all_out['confidences'],
        "targets": all_out['target_seqs'],
        "syns": all_out['target_syns'],
        "matches": [list(set(p).intersection(t)) for (p, t) in zip(all_out['predictions'], all_out['target_syns'])],
        "revealed": [p] * len(all_out['input_seqs']),
        "attns": all_out['attn_seqs']},
        columns=['ids', 'preverbs', 'targets', 'syns', 'predictions', 'matches', 'confidences', 'attns', 'revealed'])

    df = apply_inplace(df, 'preverbs',
                       lambda x: " ".join([s_index2word[w] if w in s_index2word.keys() else '<UNK>' for w in x]))
    df = apply_inplace(df, 'predictions',
                       lambda x: " ".join([t_index2word[v] for v in x]))
    df = apply_inplace(df, 'matches',
                       lambda x: " ".join([t_index2word[v] for v in x]))
    df = apply_inplace(df, 'targets',
                       lambda x: t_index2word[x])
    df = apply_inplace(df, 'syns',
                       lambda x: " ".join([t_index2word[v] for v in x]))
    df = apply_inplace(df, 'confidences',
                       lambda x: [round(c, 4) for c in x])
    df = apply_inplace(df, 'attns',
                       lambda x: [round(c, 4) for c in x if x != None])

    sorted = df.sort_values(by='ids')
    print("Writing attention samples to file...")
    sorted.to_csv(os.path.join(outfile_dir, file))


def get_cm(true_seq, pred_seq, target_dict):
    num_target = len(target_dict)
    cm = np.zeros([num_target, num_target])
    for i in range(len(true_seq)):
        pindices = [i for i, x in enumerate(pred_seq[i]) if x == 1]
        tindices = [i for i, x in enumerate(true_seq[i]) if x == 1]
        for r in tindices:
            cm[r, pindices] += 1
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(num_target):
        temp = np.delete(cm, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    pr_df = pd.DataFrame({'TP': TP,
                          'FP': FP,
                          'FN': FN,
                          'TN': TN,
                          'precision': precision,
                          'recall': recall})
    pr_df.rename(index=target_dict, inplace=True)
    return pr_df


def get_pr_multi(true_seq, pred_seq, t_word2index, output_dir, file):
    """get precision and recall table"""
    from itertools import chain
    t_index2word = {i: v for v, i in t_word2index.items()}
    true_flat = []
    pred_flat = []
    for i in range(len(true_seq)):
        if len(true_seq[i]) < len(pred_seq[i]):
            true_flat.extend(true_seq[i])

    pr_df = get_cm(true_seq, pred_seq, t_index2word)
    # report_labels = list(t_index2word.values())
    # report = classification_report(true_flat, pred_flat, labels=list(t_index2word.values()),
    #                                target_names=list(t_index2word.values()))
    pr_df.sort_values(['precision'], ascending=[0]).to_csv(os.path.join(output_dir, file))


def report2dict(cr):
    """transfer classification report to dict"""
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def get_pr_info(true_seq, pred_seq, t_word2index, output_dir, file):
    """get precision and recall table"""
    t_index2word = {i: v for v, i in t_word2index.items()}
    true_flat = [t_index2word[item] for item in true_seq]
    pred_flat = [t_index2word[item] for item in pred_seq]

    conf_mat = confusion_matrix(true_flat, pred_flat, labels=list(t_index2word.values()))
    # report_labels = list(t_index2word.values())
    report = classification_report(true_flat, pred_flat, labels=list(t_index2word.values()),
                                   target_names=list(t_index2word.values()))
    pd.DataFrame(report2dict(report)).transpose().sort_values(['f1-score'], ascending=[0]).to_csv(
        os.path.join(output_dir, file))


def creat_output_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def save_output(out_dir, file_name, obs):
    creat_output_dir(out_dir)

    with open(os.path.join(out_dir, file_name + '.pickle'), 'wb') as outfile:
        pickle.dump(obs, outfile)
    outfile.close()


def save_acc_df(acc_over_time, opt):
    creat_output_dir(opt.out_dir)
    filename = str(opt.out_dir)[2:-1] + '_acc.csv'

    p_list = list(acc_over_time.keys())
    acc = [acc_over_time[p] for p in sorted(p_list)]

    if opt.lang == 'ja_50':
        language = 'Japanese'
    else:
        language = 'German'

    df = pd.DataFrame({'revealed': sorted(p_list),
                       'acc': acc,
                       'lang': language,
                       'mode': opt.lang_mode})
    df.to_csv(os.path.join(opt.out_dir, filename))


def make_weights_for_balanced_classes(pairs, nclasses):
    count = [0] * nclasses
    for pair in pairs:
        count[pair[-1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(pairs)
    for idx, val in enumerate(pairs):
        weight[idx] = weight_per_class[val[-1]]
    return weight


class DictObj:
    def __init__(self, in_dict: dict):
        self.in_dict = in_dict
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

    def get(self, key):
        if key not in self.in_dict:
            return None
        else:
            return self.in_dict[key]

