from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random


import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaModel,
                          RobertaTokenizer)

from xglue.model import Model

import multiprocessing

import csv
import json
import sys
from io import open
from sklearn.metrics import f1_score, precision_score, recall_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label
        self.idx = idx


class InputFeaturesTriplet(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTriplet, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer, args):
    # label
    label = js['label']

    # code
    code = js['code']
    code_tokens = tokenizer.tokenize(code)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = js['doc']  # query
    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, type=None):
        # json file: dict: idx, query, doc, code
        self.examples = []
        self.type = type
        data=[]
        with open(file_path, 'r') as f:
            data = json.load(f)
        if self.type == 'test':
            for js in data:
                js['label'] = 0
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids),\
               torch.tensor(self.examples[i].label)




def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    prec = precision_score(y_true=labels, y_pred=preds)
    reca = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": prec,
        "recall": reca,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "webquery":
        return acc_and_f1(preds, labels)
    if task_name == "staqc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    args.save_steps = len(train_dataloader) if args.save_steps<=0 else args.save_steps
    args.warmup_steps = len(train_dataloader) if args.warmup_steps<=0 else args.warmup_steps
    args.logging_steps = len(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    model.to(args.device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_results = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "acc_and_f1": 0.0}
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    model.train()
    logger.info(model)

    for idx in train_iterator:
        bar = tqdm(enumerate(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in bar:

            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device)
            loss, predictions = model(code_inputs, nl_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} step {} loss {}".format(idx, step+1, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))
                        # Save model checkpoint
                        if results['acc_and_f1'] >= best_results['acc_and_f1']:
                            best_results = results

                            # save
                            checkpoint_prefix = 'checkpoint-best-ever'
                            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    if args.local_rank == -1:
                        checkpoint_prefix = 'checkpoint-last'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(output_dir)

                        idx_file = os.path.join(output_dir, 'idx_file.txt')
                        with open(idx_file, 'w', encoding='utf-8') as idxf:
                            idxf.write(str(args.start_epoch + idx) + '\n')
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                        step_file = os.path.join(output_dir, 'step_file.txt')
                        with open(step_file, 'w', encoding='utf-8') as stepf:
                            stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

def evaluate(args, model, tokenizer,eval_when_training=False):
    eval_output_dir = args.output_dir
    eval_data_path = os.path.join(args.data_dir, args.dev_file)
    eval_dataset = TextDataset(tokenizer, args, eval_data_path, type='eval')

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, predictions = model(code_inputs, nl_inputs, labels)
            # lm_loss,code_vec,nl_vec = model(code_inputs,nl_inputs)
            eval_loss += lm_loss.mean().item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        nb_eval_steps += 1
    all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()
    all_labels = torch.cat(all_labels, 0).squeeze().numpy()
    eval_loss = torch.tensor(eval_loss / nb_eval_steps)

    results = acc_and_f1(all_predictions, all_labels)
    results.update({"eval_loss": float(eval_loss)})
    return results


def test(args, model, tokenizer):
    if not args.prediction_file:
        args.prediction_file = os.path.join(args.output_dir, 'predictions.txt')
    if not os.path.exists(os.path.dirname(args.prediction_file)):
        os.makedirs(os.path.dirname(args.prediction_file))

    test_data_path = os.path.join(args.data_dir, args.test_file)
    eval_dataset = TextDataset(tokenizer, args, test_data_path, type='test')

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nb_eval_steps = 0
    all_predictions = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            _, predictions = model(code_inputs, nl_inputs, labels)
            all_predictions.append(predictions.cpu())
        nb_eval_steps += 1
    all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()

    logger.info("***** Running Test *****")
    with open(args.prediction_file,'w') as f:
        for example, pred in zip(eval_dataset.examples, all_predictions.tolist()):
            f.write(example.idx+'\t'+str(int(pred))+'\n')

