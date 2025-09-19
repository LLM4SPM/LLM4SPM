# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import re
import sys
import code2nl_cross.bleu as bleu
import code2nl_cross.bleu as model_new
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from code2nl_cross.model_1 import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartForConditionalGeneration, BartConfig,
                          T5ForConditionalGeneration, T5Config, T5Tokenizer,
                          AutoModel, AutoTokenizer, AutoConfig,
                          LlamaModel, LlamaTokenizer, LlamaConfig,PLBartTokenizer, PLBartForConditionalGeneration, PLBartConfig,T5ForConditionalGeneration)

# Extended model classes dictionary - now supporting unified encoder-decoder models
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),  # CodeBERT uses RoBERTa architecture
    'llama': (LlamaConfig, LlamaModel, LlamaTokenizer),
    'codellama': (AutoConfig, LlamaModel, AutoTokenizer),  # CodeLlama uses Llama architecture
    'bart': (PLBartConfig,PLBartForConditionalGeneration,PLBartTokenizer),  # BART as unified model
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),  # T5 as unified model
    'auto': (AutoConfig, AutoModel, AutoTokenizer),  # Generic auto loader for any model
    'codereviewer':(T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

root = os.path.dirname(__file__)

import nltk
from datasets import load_metric


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

#commitbart
#def read_examples(filename):
#     examples = []
#     with open(filename, encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             js = json.loads(line)
#             if 'idx' not in js:
#                 js['idx'] = idx
# 
#             source = ' '.join(js.get('code_tokens', []))
#             target = ' '.join(js.get('docstring_tokens', []))
# 
#             examples.append(Example(
#                 idx=js['idx'],
#                 source=' '.join(source.strip().split()),
#                 target=' '.join(target.strip().split()),
#             ))
#     return examples

def read_examples(filename):
   """Read examples from filename."""
   examples = []
   with open(filename, encoding="utf-8") as f:
       for idx, line in enumerate(f):
           line = line.strip()
           js = json.loads(line)
           if 'idx' not in js:
               js['idx'] = idx

            #Combine different source fields if available
           source_parts = []
           if 'contents' in js:
               contents = ' '.join([str(x) for x in js['contents'] if x is not None]).replace('\n', ' ')
               source_parts.append(contents)
           if 'code_tokens' in js:
             code = ' '.join([str(x) for x in js['code_tokens'] if x is not None]).replace('\n', ' ')
             code = ' '.join(code.strip().split())
             source_parts.append(code)
           elif 'code' in js:
             code = js['code']
             if code is not None:
                 code = code.replace('\n', ' ')
                 code = ' '.join(code.strip().split())
                 source_parts.append(code)
         # Join all source parts
           source = ' '.join(source_parts)

# Get target text
           if 'docstring_tokens' in js:
               nl = ' '.join(js['docstring_tokens']).replace('\n', '')
           elif 'docstring' in js:
               nl = js['docstring'].replace('\n', '')
           elif 'summary' in js:
               nl = js['summary'].replace('\n', '')
           else:
               nl = ''

           nl = ' '.join(nl.strip().split())
           examples.append(
              Example(
                  idx=idx,
                  source=source,
                  target=nl,
              )
          )
   return examples
class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # Source encoding
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # Target encoding
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))
                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_dir = root + '/output/'


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='t5', type=str,
                        help="Model type: e.g. roberta, codebert, llama, codellama, bart, t5, auto")
    parser.add_argument("--model_name_or_path", default='/home/linxw/yzu-rza/vulfixminer1/vulfixminer/codet5', type=str,
                        help="Path to pre-trained model: e.g. roberta-base, microsoft/codebert-base, codellama/CodeLlama-7b-hf, facebook/bart-base, t5-base")
    parser.add_argument("--decoder_model_name_or_path", default=None, type=str,
                        help="Path to pre-trained decoder model (only needed for encoder-only models): e.g. facebook/bart-base, t5-base")
    parser.add_argument("--output_dir", default=root + '/result/key_aspect/AV_code_msg_codet5', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default='data/code_msg/AV_train_file.jsonl', type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default='data/code_msg/AV_test_file.jsonl', type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=128, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=4, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    def clean_generated_text(text, tokenizer):
        """清理生成的文本"""
        # 移除特殊标记
        special_tokens = [
            tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
            "<s>", "</s>", "<pad>", "<unk>"
        ]
        for token in special_tokens:
            text = text.replace(token, "")

        # 移除占位符
        text = re.sub(r"<extra_id_\d+>", "", text)

        # 清理多余的空格和符号
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[^a-zA-Z0-9]+", "", text)  # 移除开头的非字母数字字符
        text = re.sub(r"[^a-zA-Z0-9\.\?\!]+$", "", text)  # 移除结尾的非字母数字字符

        # 处理多余的括号
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        return text
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # Load model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # Load the unified model
    unified_model = model_class.from_pretrained(args.model_name_or_path, config=config)

    # Resize token embeddings if necessary
    unified_model.resize_token_embeddings(len(tokenizer))

    # Build complete model
    model = Seq2Seq(model=unified_model,
                    config=config,
                    tokenizer=tokenizer,
                    beam_size=args.beam_size,
                    max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id,
                    eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        metric = load_metric("rouge", trust_remote_code=True)

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        def compute_metrics(preds, labels):
            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(preds, labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        logger.info("  output = %s", args.output_dir)
        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        best_rouge = 0

        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                             target_ids=target_ids, target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Print loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                    # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(str(gold.idx) + '\t' + ref + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target + '\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

        # Save final model
        output_dir = os.path.join(args.output_dir, 'checkpoint-final')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        if args.do_test:
            files = []
            if args.dev_filename is not None:
                files.append(args.dev_filename)
            if args.test_filename is not None:
                files.append(args.test_filename)

            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)

            for idx, file in enumerate(files):
                logger.info("Test file: {}".format(file))
                eval_examples = read_examples(file)
                eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                p = []

                # 添加生成代码开始
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch

                    with torch.no_grad():
                        outputs = model(source_ids=source_ids, source_mask=source_mask)

                        for output in outputs:
                            tokens = output.view(-1).tolist()
                            # 移除填充部分
                            tokens = [token for token in tokens if token != tokenizer.pad_token_id]
                            text = tokenizer.decode(tokens, skip_special_tokens=False)

                            # 增强后处理
                            text = clean_generated_text(text, tokenizer)
                            p.append(text)
                # 添加生成代码结束

                # 只在主进程中写入文件
                if args.local_rank in [-1, 0]:
                    predictions = []
                    rouge_pred, rouge_gold = [], []

                    # 修复路径格式
                    output_file = os.path.join(args.output_dir, f"test_{idx}.output")
                    gold_file = os.path.join(args.output_dir, f"test_{idx}.gold")
                    result_file = os.path.join(args.output_dir, f"result_{idx}.output")

                    try:
                        with open(output_file, 'w', encoding='utf-8') as f, \
                                open(gold_file, 'w', encoding='utf-8') as f1, \
                                open(result_file, 'w', encoding='utf-8') as f2:

                            for ref, gold in zip(p, eval_examples):
                                line = f"{gold.idx}\t{ref}"
                                predictions.append(line)
                                f.write(line + '\n')
                                f1.write(f"{gold.idx}\t{gold.target}\n")
                                rouge_pred.append(ref)
                                rouge_gold.append(gold.target)

                            # 计算并写入ROUGE结果
                            matrix = compute_metrics(rouge_pred, rouge_gold)
                            f2.write(json.dumps(matrix, indent=2))

                        logger.info(f"Successfully saved files: {output_file}, {gold_file}, {result_file}")

                    except Exception as e:
                        logger.error(f"Error writing files: {str(e)}")

                # 计算BLEU（所有进程都计算但只主进程输出）
                if args.local_rank in [-1, 0]:
                    try:
                        (goldMap, predictionMap) = bleu.computeMaps(predictions, gold_file)
                        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 5)
                        logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                        logger.info("  " + "*" * 20)
                        # === 新增：保存到独立文件 ===
                        bleu_file = os.path.join(args.output_dir, f"bleu_{idx}.txt")
                        with open(bleu_file, "w", encoding="utf-8") as f_bleu:
                            f_bleu.write(f"BLEU-4: {dev_bleu}\n")
                        logger.info(f"BLEU-4 saved to {bleu_file}")

                        # === 新增：合并写入 result_{idx}.output（之前已写了ROUGE）===
                        result_path = os.path.join(args.output_dir, f"result_{idx}.output")
                        try:
                            with open(result_path, "r", encoding="utf-8") as fr:
                                result_json = json.load(fr)
                        except Exception:
                            result_json = {}
                        result_json["BLEU-4"] = dev_bleu
                        with open(result_path, "w", encoding="utf-8") as fw:
                            json.dump(result_json, fw, ensure_ascii=False, indent=2)
                        logger.info(f"BLEU-4 merged into {result_path}")
                    except Exception as e:
                        logger.error(f"Error computing BLEU: {str(e)}")
    # 修改文件写入部分，添加主进程判断和路径修复
if __name__ == "__main__":
    main()