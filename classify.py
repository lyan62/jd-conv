import yaml
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import torch
import numpy as np
import os
import models
from utils.utils import *
from dataset.querySet import *
import multiprocessing
import logging as log
import time
import argparse

def calc_loss(loss_func, prediction, target_batches):
    if loss_func == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        return criterion(prediction, target_batches.view(-1))
    elif loss_func == 'max_margin':
        criterion = nn.MultiMarginLoss()
        return criterion(prediction, target_batches.view(-1))


def iter(batched_variables, lengths, predictor, hidden, train_flag, args, argsimizer=None):
    if train_flag =='TRAIN':
        predictor.train()  # init train mode
        predictor.zero_grad()  # clear the gradients before loss and backward
    else:
        predictor.eval()

    if batched_variables['input'].size(0) != args.batch_size:
        hidden = predictor.init_hidden(batched_variables['input'].size(0))
    else:
        hidden = repackage_hidden(hidden, reset=True)

    # make prediction
    if args.apply_attn:
        prediction, scores, hidden = predictor(batched_variables, lengths, hidden)
    else:
        prediction, hidden = predictor(batched_variables, lengths, hidden)

    loss = calc_loss(args.loss_func, prediction, batched_variables['target'])

    if train_flag == 'TRAIN':
        loss.backward()
        argsimizer.step()

    conf, out = F.softmax(prediction, dim=1).topk(1)
    target = batched_variables['target'].data.view(-1)

    output_dict = {'pred':out.data.view(-1),
                   'conf':conf.data.view(-1),
                   'target':target,
                   }

    if args.apply_attn:
        output_dict['attn'] = scores

    return loss.data.item(), output_dict

def train_epoch(train_set, eval_set, predictor, argsimizer, epoch, val_history_acc, val_history_loss, scheduler, args):
    start_time = time.time()  # track time
    total_loss = 0  # track loss
    batch_count = 0  # track batch

    correct = 0  # consider only top1
    total = 0
    history_acc = []

    total_count = train_set.__len__()

    if args.sampler == 'random':
        data_sampler = torch.utils.data.sampler.RandomSampler(train_set)
        shuffling = False
    else:
        data_sampler = None
        shuffling = args.shuffling

    # set up data loader
    data_loader = DataLoader(train_set, args.batch_size, shuffle=shuffling, collate_fn=my_collate_fn,
                             num_workers=4, pin_memory=True, sampler=data_sampler)

    hidden = predictor.init_hidden(args.batch_size) # init hidden

    for batched_data, lengths in data_loader:
        batched_variables = {k: variableFromSentence(v) for k, v in batched_data.items() if k!="text"}
        # input variable( B x L), pos_variable ( B X L ), target_variable(1 x B)

        loss, output = iter(batched_variables, lengths, predictor, hidden, 'TRAIN', args, argsimizer=argsimizer)

        total_loss += loss
        batch_count += 1

        correct += torch.sum(torch.eq(output['pred'], output['target'])).item()

        total += output['target'].size(0)
        acc = correct / total
        history_acc.append(acc)

        # log
        if batch_count % args.log_interval == 0 and batch_count > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            log.info('{:5d}/{:5d} batches |ms/batch {:5.2f} | loss {:5.4f} | acc {:5.4f}'
                  .format(batch_count, total_count // args.batch_size, elapsed * 1000 / args.log_interval, cur_loss, acc))
            total_loss = 0
        start_time = time.time()

        # eval
        if batch_count % args.eval_interval == 0 and batch_count > 0:
            val_loss, val_acc = eval(eval_set, predictor, args)  # eval

            # Save checkpoint if is a new best
            is_best = bool(val_acc > max(val_history_acc))
            if is_best:
                log.info("Getting a new best...")
                predictor.save(args.ckp_dir)

            print_progress(epoch, batch_count, val_loss)

            val_history_acc.append(val_acc)
            val_history_loss.append(val_loss)
            scheduler.step(val_loss)  # reduce learning rate if val_loss stops decreasing


    return history_acc, val_history_acc, val_history_loss


def eval(eval_set, predictor, args):
    predictor.eval()
    total_loss = 0
    batch_count = 0
    correct = 0
    total = 0

    # get dataset
    batch_size = args.batch_size
    data_loader = DataLoader(eval_set, batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=4,
                             pin_memory=True)

    hidden = predictor.init_hidden(batch_size)
    for batched_data, lengths in data_loader:
        batched_variables = {k: variableFromSentence(v) for k, v in batched_data.items() if k!='text'}

        loss, output = iter(batched_variables, lengths, predictor, hidden,'EVAL',args)

        total_loss += loss
        batch_count += 1

        total += output['target'].size(0)
        correct += torch.sum(torch.eq(output['pred'], output['target'])).item()


    avg_loss = total_loss / batch_count  # average loss
    avg_acc = float(correct / total)
    log.info("Test average loss:%.4f, overall accuracy %.4f" % (avg_loss, float(correct / total)))

    return avg_loss, avg_acc


def train(args):

    # initiliaze models
    train_set = querySet(args.train_data)
    print("total num of queries in train: %d" % train_set.__len__())
    if os.path.exists("./output/dict.pkl"):
        dictionary = pickle.load(open(os.path.join("./output/", "dict.pkl"), "rb"))
    else:
        dictionary = build_dict(train_set.df)
        creat_output_dir("output")
        pickle.dump(dictionary, open(os.path.join("./output/", "dict.pkl"), "wb"))

    eval_set = querySet(args.eval_data)

    predictor = getattr(models, args.model)(dictionary, args)
    if use_cuda:
        predictor = predictor.cuda()

    # initialize argsimizer
    argsimizer = optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=1e-5)
    # initialize scheduler
    scheduler = ReduceLROnPlateau(argsimizer, 'min', patience=2, verbose=True)

    train_history_acc = []
    val_history_acc = [0.0]
    val_history_loss = [1000]


    if args.load_ckp:
        predictor.load(args.ckp_dir)

    for epoch in range(1, args.num_epochs + 1):

        train_acc, val_history_acc, val_history_loss = train_epoch(train_set, eval_set, predictor, argsimizer,
                                                 epoch, val_history_acc, val_history_loss, scheduler, args) # train

        train_history_acc.extend(train_acc)

    save_output(args.out_dir, 'train_acc', train_history_acc)
    save_output(args.out_dir, 'eval_acc', val_history_acc)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", dest="train", action='store_true', help="train model")
    argparser.add_argument("--config", dest="config", action='store', type=str)
    params = argparser.parse_args()

    # enable multiprocessing
    multiprocessing.get_context('spawn')
    # use gpu
    use_cuda = torch.cuda.is_available()

    # setting up log
    logdatetime = time.strftime("%m%d")
    format = '  %(message)s'
    handlers = [log.FileHandler('train'+logdatetime+'.log'), log.StreamHandler()]
    log.basicConfig(level=log.INFO, format=format, handlers=handlers)
    # parse parameters using yaml
    config_file = params.config
    with open(config_file) as input_file:
        config= yaml.load(input_file, Loader=yaml.FullLoader)

    args = DictObj(config)
    train_args = args.train
    # print(train_args.get("cnn_rnn"))

    if params.train:
        train(train_args)
