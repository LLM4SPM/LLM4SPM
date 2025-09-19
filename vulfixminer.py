 # -*- coding: UTF-8 -*-
import os
import tiktoken
#os.environ["TIKTOKEN_CACHE_DIR"] = "/home/linxw/yzu-rza/vulfixminer1/vulfixminer/Dataset"

#enc = tiktoken.get_encoding("gpt2")
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW,PLBartConfig,AutoConfig
from transformers import get_scheduler
from patch_entities import VulFixMinerDataset
from model import VulFixMinerClassifier, VulFixMinerFineTuneClassifier,VulFixMinerFineTuneClassifier_CommitBART,VulFixMinerFineTuneClassifier_CodeGen25_7b
import pandas as pd
from tqdm import tqdm
# import utils
import config
import argparse
import vulfixminer_finetune
import vulfixminer_commitbart_finetune
import vulfixminer_codegen_finetune
from transformers import RobertaTokenizer, RobertaModel, logging,AutoTokenizer, AutoModelForCausalLM
import csv 
from evaluation import cost_effort_at_L, popt_at_L,calculate_fpr
logging.set_verbosity_error()
# dataset_name = 'sap_patch_dataset.csv'
# EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2'
# MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'

dataset_name = None
FINETUNE_MODEL_PATH = None
MODEL_PATH = None
TRAIN_PROB_PATH = None
TEST_PROB_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(directory, 'model')
#NUMBER_OF_EPOCHS = 20
# NUMBER_OF_EPOCHS = 10
EARLY_STOPPING_ROUND = 5

#TRAIN_BATCH_SIZE = 64
#VALIDATION_BATCH_SIZE = 64
#TEST_BATCH_SIZE = 64

# retest with SAP dataset
NUMBER_OF_EPOCHS = 5
# NUMBER_OF_EPOCHS = 10
#EARLY_STOPPING_ROUND = 5

TRAIN_BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8

# TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
# VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
# TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}


# LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-5
# LEARNING_RATE = 1e-7

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 128
HIDDEN_DIM = 4096

NUMBER_OF_LABELS = 2


# model_path_prefix = model_folder_path + '/patch_variant_2_16112021_model_'
def custom_collate_fn(batch):

    ids = []
    urls = []
    embeddings = []
    labels = []
    
    for item in batch:
        id_, url, embedding, label = item
        ids.append(id_)
        urls.append(url)

        if isinstance(embedding, torch.Tensor):
            embeddings.append(embedding)
        else:
            try:
                embeddings.append(torch.tensor(embedding, dtype=torch.float32))
            except Exception as e:
                print(f"Error converting embedding to tensor: {e}")
                embeddings.append(torch.zeros(768, dtype=torch.float32))

        if isinstance(label, (int, float)):
            labels.append(label)
        elif isinstance(label, str):
            try:
                labels.append(int(label))
            except ValueError:
                print(f"Invalid label value: {label}")
                labels.append(0)  
        else:
            labels.append(0)  

    try:
        embedding_tensor = torch.stack(embeddings)
        label_tensor = torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        print(f"Error stacking tensors: {e}")
        return [], [], torch.tensor([]), torch.tensor([])
    
    return ids, urls, embedding_tensor, label_tensor


def predict_test_data(model, testing_generator, device, need_prob=False, need_feature_only=False, prob_path=None):
    y_pred = []
    y_test = []
    probs = []
    urls = []
    final_features = []
    with torch.no_grad():
        model.eval()
        for ids, url_batch, embedding_batch, label_batch in tqdm(testing_generator):
            if not isinstance(embedding_batch, torch.Tensor):
                print(f"Invalid embedding_batch type: {type(embedding_batch)}, sample: {embedding_batch[:2]}")
                continue
            if not isinstance(label_batch, torch.Tensor):
                print(f"Invalid label_batch type: {type(label_batch)}, sample: {label_batch[:2]}")
                continue
            try:
                embedding_batch, label_batch = embedding_batch.to(device), label_batch.to(device)
                outs = model(embedding_batch)
                if need_feature_only:
                    final_features.extend(outs[1].tolist())
                    outs = outs[0]

                outs = F.softmax(outs, dim=1)

                y_pred.extend(torch.argmax(outs, dim=1).tolist())
                y_test.extend(label_batch.tolist())
                probs.extend(outs[:, 1].tolist())
                urls.extend(url_batch)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test, average='macro')
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test, average='macro')
        mcc=metrics.matthews_corrcoef(y_pred=y_pred, y_true=y_test)
        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")

    if prob_path is not None:
        with open(prob_path, 'w') as file:
            writer = csv.writer(file)
            for i, prob in enumerate(probs):
                writer.writerow([urls[i], prob])


    if need_feature_only:
        return f1, urls, final_features

    if not need_prob:
        return precision, recall, f1, auc,mcc
    else:
        return precision, recall, f1, auc,mcc, y_test, probs,y_pred


def train(model, learning_rate, number_of_epochs, training_generator, test_generator,loc_list):
    metrics_list = []
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []  
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, url_batch, embedding_batch, label_batch in training_generator:
            #embedding_batch, label_batch \
               # = embedding_batch.to(device), label_batch.to(device)
            embedding_batch = embedding_batch.to(device, dtype=torch.float32)  # 显式转换为 float32
            label_batch = label_batch.to(device, dtype=torch.long)  #
            outs = model(embedding_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        print("Result on testing dataset...")
        precision, recall, f1, auc,mcc,y_test, probs,y_pred = predict_test_data(model=model,
                                                       testing_generator=test_generator,
                                                       device=device,need_prob=True)
        print("Vulnerability fix ratio:", np.mean(y_test))
        #popt_val = popt(y_test, probs, loc_list)
        metrics_dict={
             "epoch": epoch,
             "precision": precision,
             "recall": recall,
             "f1": f1,
             "auc":auc,
             "mcc":mcc}
        if  "java" in dataset_name.lower() or "python" in dataset_name.lower():
            # 计算 CostEffort@5 和 @20
            ce5 = cost_effort_at_L(y_test, probs, loc_list, L=5)
            ce10 = cost_effort_at_L(y_test, probs, loc_list, L=10)
            ce15 = cost_effort_at_L(y_test, probs, loc_list, L=15)
            ce20 = cost_effort_at_L(y_test, probs, loc_list, L=20)
            # 计算 Popt
            popt_5 = popt_at_L(y_test, probs, loc_list, L=5)
            popt_20 = popt_at_L(y_test, probs, loc_list, L=20)
            metrics_dict.update({
                'ce5': ce5,
                'ce10':ce10,
                'ce15':ce15,
                'ce20': ce20,
                'popt5': popt_5,
                'popt20': popt_20
            })
            print(f"CostEffort@5%: {ce5:.2f}%,CostEffort@5%: {ce10:.2f}%,CostEffort@5%: {ce15:.2f}%, CostEffort@20%: {ce20:.2f}%")
            print(f"Popt@5%: {popt_5:.4f}")
            print(f"Popt@20%: {popt_20:.4f}")
        elif "output1" in dataset_name.lower():
            fpr = calculate_fpr(y_test, y_pred)
            metrics_dict.update({'fpr': fpr})
            print(f"False Positive Rate: {fpr:.4f}")
        
        metrics_list.append(metrics_dict)
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

    # 动态确定 CSV 字段
    fieldnames = ['epoch', 'precision', 'recall', 'f1', 'auc','mcc']
    if  "java" in dataset_name.lower() or "python" in dataset_name.lower():
        fieldnames.extend(['ce5','ce10','ce15', 'ce20', 'popt5', 'popt20'])
    elif "output1" in dataset_name.lower():
        fieldnames.append('fpr')
    
    metrics_path = MODEL_PATH.replace('.sav', '_metrics.csv')
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics_list:
            writer.writerow(metric)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), MODEL_PATH)
    else:
        torch.save(model.state_dict(), MODEL_PATH)

    return model

class CommitAggregator:
    def __init__(self, file_transformer):
        # self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        tokenizer = AutoTokenizer.from_pretrained(
            "./CodeLlama-7b-hf",
            padding_side="left",
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        # self.tokenizer = RobertaTokenizer.from_pretrained("./graphcodebert")
        self.file_transformer = file_transformer

    def transform(self, diff_list):
        # cap at 20 diffs
        diff_list = diff_list[:5]
        input_list, mask_list = [], []
        for diff in diff_list:
            added_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=True)
            deleted_code = vulfixminer_finetune.get_code_version(diff=diff, added_version=False)
            code = added_code + self.tokenizer.eos_token + deleted_code
            # code = added_code + self.tokenizer.sep_token + deleted_code
            input_ids, mask = vulfixminer_finetune.get_input_and_mask(self.tokenizer, [code])
            input_list.append(input_ids)
            mask_list.append(mask)

        input_list = torch.stack(input_list)
        mask_list = torch.stack(mask_list)
        input_list, mask_list = input_list.to(device), mask_list.to(device)
        # embeddings = self.file_transformer(input_list, mask_list).last_hidden_state[:, 0, :]
        outputs = self.file_transformer(
            input_ids=input_list,
            attention_mask=mask_list,
            output_hidden_states=True  # 关键参数
        )

        # 获取最后一层隐藏状态
        last_hidden_states = outputs.hidden_states[-1]
        # 获取实际序列结束位置的索引
        last_token_indices = mask_list.sum(dim=1) - 1
        # 提取对应位置的embedding
        embeddings = last_hidden_states[torch.arange(input_list.size(0)), last_token_indices]
        sum_ = torch.sum(embeddings, dim=0)
        mean_ = torch.div(sum_, len(diff_list))
        mean_ = mean_.detach()
        mean_ = mean_.cpu()

        return mean_
#commitbart
#class CommitAggregator:
#     def __init__(self, file_transformer):
#         # 加载 CommitBART 的分词器
#         tokenizer = AutoTokenizer.from_pretrained(
#             "./CommitBART",
#             padding_side="right",
#             use_fast=True
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#         self.tokenizer = tokenizer
#         self.file_transformer = file_transformer
# 
#     def transform(self, diff_list):
#         # 限制最多处理 5 个 diff
#         diff_list = diff_list[:5]
#         input_list, mask_list = [], []
# 
#         for diff in diff_list:
#             print(f"Processing diff: {diff[:50]}... (type: {type(diff)})")
#             added_code = vulfixminer_commitbart_finetune.get_code_version(diff=diff, added_version=True)
#             deleted_code = vulfixminer_commitbart_finetune.get_code_version(diff=diff, added_version=False)
#             code = added_code + self.tokenizer.sep_token + deleted_code
#             input_ids, mask = vulfixminer_commitbart_finetune.get_input_and_mask(self.tokenizer, [code])
#             input_list.append(input_ids)
#             mask_list.append(mask)
# 
#         # 转换为张量
#         input_list = torch.stack(input_list)
#         mask_list = torch.stack(mask_list)
#         input_list, mask_list = input_list.to(device), mask_list.to(device)
# 
#         outputs = self.file_transformer(
#             input_ids=input_list,
#             attention_mask=mask_list,
#             return_dict=True
#         )
#         last_hidden_state = outputs.last_hidden_state
#         embeddings = last_hidden_state.mean(dim=1)
#         sum_ = torch.sum(embeddings, dim=0)
#         mean_ = torch.div(sum_, len(diff_list))
#         mean_ = mean_.to(dtype=torch.float32)
#         mean_ = mean_.detach()
#         mean_ = mean_.cpu()
# 
#         return mean_

# class CommitAggregator:
#     def __init__(self, file_transformer):
#         # 加载 CodeGen2.5-7B 的分词器
#         tokenizer = AutoTokenizer.from_pretrained(
#             "./CodeGen25_7b_mono",
#             padding_side="left",
#             use_fast=True,
#             trust_remote_code=True
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
#             tokenizer.pad_token_id = tokenizer.eos_token_id or 50256
#         if tokenizer.sep_token is None:
#             tokenizer.sep_token = tokenizer.eos_token  # 使用 <|endoftext|> 作为分隔符
#             tokenizer.sep_token_id = tokenizer.eos_token_id
#         self.tokenizer = tokenizer
#         self.file_transformer = file_transformer
# 
#     def transform(self, diff_list):
#         # 限制最多处理 5 个 diff
#         diff_list = diff_list[:5]
#         input_list, mask_list = [], []
# 
#         for diff in diff_list:
#             added_code = vulfixminer_codegen_finetune.get_code_version(diff=diff, added_version=True)
#             deleted_code = vulfixminer_codegen_finetune.get_code_version(diff=diff, added_version=False)
#             code = added_code + self.tokenizer.sep_token + deleted_code
#             #code = added_code +  deleted_code
#             input_ids, mask = vulfixminer_codegen_finetune.get_input_and_mask(self.tokenizer, [code])
#             input_list.append(input_ids)
#             mask_list.append(mask)
#         # 转换为张量
#         input_list = torch.stack(input_list)
#         mask_list = torch.stack(mask_list)
#         input_list, mask_list = input_list.to(device), mask_list.to(device)
#         outputs = self.file_transformer(
#               input_ids=input_list,
#               attention_mask=mask_list,
#               output_hidden_states=True  # 关键参数
#                )
#         last_hidden_states = outputs.hidden_states[-1]
#         last_token_indices = mask_list.sum(dim=1) - 1
#         embeddings = last_hidden_states[torch.arange(input_list.size(0)), last_token_indices]
#         sum_ = torch.sum(embeddings, dim=0)
#         mean_ = torch.div(sum_, len(diff_list))
#         mean_ = mean_.detach()
#         mean_ = mean_.cpu()
#         return mean_

def do_train(args):
    global dataset_name, MODEL_PATH
    loc_list = []
    dataset_name = args.dataset_path
    FINETUNE_MODEL_PATH = args.finetune_model_path
    MODEL_PATH = args.model_path
    TRAIN_PROB_PATH = args.train_prob_path
    TEST_PROB_PATH = args.test_prob_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(MODEL_PATH))

    print("Loading finetuned file transformer...")
    finetune_model = VulFixMinerFineTuneClassifier()
    #finetune_model = VulFixMinerFineTuneClassifier_CommitBART()
    #finetune_model = VulFixMinerFineTuneClassifier_CodeGen25_7b()
    #if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #finetune_model = nn.DataParallel(finetune_model)
    finetune_model=finetune_model.to(device)
    finetune_model.load_state_dict(torch.load(FINETUNE_MODEL_PATH), False)
    # code_bert = finetune_model.module.code_bert
    code_llama = finetune_model.code_llama
    code_llama.eval()
    code_llama.to(device)
    #commit_bart = finetune_model.commit_bart
    #commit_bart.eval()
    #commit_bart.to(device)
    #code_gen = finetune_model.code_gen
    #code_gen.eval()
    #code_gen.to(device)
    print("Finished loading")

    aggregator = CommitAggregator( code_llama)

    patch_data, label_data, url_data = vulfixminer_finetune.get_data(dataset_name)
    #patch_data, label_data, url_data = vulfixminer_finetune.get_data(dataset_name)
    print("patch_data type:", type(patch_data['train'][0]), patch_data['train'][0][:2])  # 检查前两个 diff
    print("label_data type:", type(label_data['train'][0]))
    print("url_data type:", type(url_data['train'][0]))
    for diff_list in patch_data['test']:
        total_loc = 0
        for diff in diff_list:
            added = vulfixminer_finetune.get_code_version(diff, added_version=True)
            deleted = vulfixminer_finetune.get_code_version(diff, added_version=False)
            total_loc += added.count('\n') + deleted.count('\n')
        loc_list.append(total_loc)
    plt.hist(loc_list, bins=20)
    plt.xlabel('LOC')
    plt.ylabel('Frequency')
    plt.title('LOC Distribution')
    plt.savefig('loc_distribution.png')
    plt.show()
    print("loc_list[:10]:", loc_list[:10])
    print("Unique LOC values:", np.unique(loc_list))
    print("Min LOC:", min(loc_list), "Max LOC:", max(loc_list), "Mean LOC:", np.mean(loc_list))
    train_ids, test_ids = [], []

    index = 0

    id_to_embeddings, id_to_label, id_to_url = {}, {}, {}
    for i in tqdm(range(len(patch_data['train']))):
        label = label_data['train'][i]
        url = url_data['train'][i]
        embeddings = aggregator.transform(patch_data['train'][i])
        train_ids.append(index)
        id_to_embeddings[index] = embeddings
        id_to_label[index] = label
        id_to_url[index] = url
        # all_data.append(embeddings)
        # all_label.append(label)
        # all_url.append(url)
        index += 1

    for i in tqdm(range(len(patch_data['test']))):
        label = label_data['test'][i]
        url = url_data['test'][i]
        embeddings = aggregator.transform(patch_data['test'][i])
        test_ids.append(index)
        id_to_embeddings[index] = embeddings
        id_to_label[index] = label
        id_to_url[index] = url
        # all_data.append(embeddings)
        # all_label.append(label)
        # all_url.append(url)
        index += 1


    training_set = VulFixMinerDataset(train_ids, id_to_label, id_to_embeddings, id_to_url)
    test_set = VulFixMinerDataset(test_ids, id_to_label, id_to_embeddings, id_to_url)
    
    #training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    #test_generator = DataLoader(test_set, **TEST_PARAMS)
    training_params = {'batch_size': TRAIN_BATCH_SIZE, 
                      'shuffle': True, 
                      'num_workers': 0,
                      'collate_fn': custom_collate_fn}
    
    test_params = {'batch_size': TEST_BATCH_SIZE, 
                   'shuffle': False, 
                   'num_workers': 0,
                   'collate_fn': custom_collate_fn}
    
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)
    #commitbart
    #config= PLBartConfig.from_pretrained("./CommitBART")
    config= PLBartConfig.from_pretrained("./CodeLlama-7b-hf")
    #config= AutoConfig.from_pretrained("./CodeGen25_7b_mono")
    model = VulFixMinerClassifier(hidden_dim=config.hidden_size)
    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_generator=test_generator,
         loc_list=loc_list)

    print("Writing result to file...")
    predict_test_data(model=model, testing_generator=training_generator, device=device, prob_path=TRAIN_PROB_PATH)
    predict_test_data(model=model, testing_generator=test_generator, device=device, prob_path=TEST_PROB_PATH)
    print("Finish writting")

def evaluate_only(args):
    global dataset_name, MODEL_PATH
    loc_list = []
    dataset_name = args.dataset_path
    FINETUNE_MODEL_PATH = args.finetune_model_path
    MODEL_PATH = args.model_path
    TEST_PROB_PATH = args.test_prob_path

    print("Dataset name: {}".format(dataset_name))
    print("Loading models...")
    
    # 加载预训练模型
    finetune_model = VulFixMinerFineTuneClassifier()
    finetune_model = finetune_model.to(device)
    
    # 检查预训练模型参数
    pretrained_state_dict = torch.load(FINETUNE_MODEL_PATH)
    print("\nPretrained model parameters:")
    print(f"Number of parameters to load: {len(pretrained_state_dict)}")
    
    # 检查模型参数匹配情况
    model_state_dict = finetune_model.state_dict()
    missing_keys = []
    unexpected_keys = []
    for key in model_state_dict:
        if key not in pretrained_state_dict:
            missing_keys.append(key)
    for key in pretrained_state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)
            
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # 使用strict=True加载预训练模型
    try:
        finetune_model.load_state_dict(pretrained_state_dict)
        print("Successfully loaded pretrained model with strict=True")
    except Exception as e:
        print(f"Error loading pretrained model with strict=True: {e}")
        print("Falling back to strict=False")
        finetune_model.load_state_dict(pretrained_state_dict, strict=False)
    
    code_llama= finetune_model.code_llama
    code_llama.eval()
    code_llama.to(device)
    
    # 加载分类器
    #config = PLBartConfig.from_pretrained("./CodeGen25_7b_mono")
    config = AutoConfig.from_pretrained("./CodeLlama-7b-hf")
    model = VulFixMinerClassifier(hidden_dim=config.hidden_size)
    model.to(device)
    
    # 检查分类器参数
    classifier_state_dict = torch.load(MODEL_PATH)
    print("\nClassifier model parameters:")
    print(f"Number of parameters to load: {len(classifier_state_dict)}")
    
    # 检查分类器参数匹配情况
    model_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    for key in model_state_dict:
        if key not in classifier_state_dict:
            missing_keys.append(key)
    for key in classifier_state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)
            
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # 使用strict=True加载分类器
    try:
        model.load_state_dict(classifier_state_dict)
        print("Successfully loaded classifier model with strict=True")
    except Exception as e:
        print(f"Error loading classifier model with strict=True: {e}")
        print("Falling back to strict=False")
        model.load_state_dict(classifier_state_dict, strict=False)
    
    model.eval()
    print("Finished loading models")

    # 数据处理
    aggregator = CommitAggregator(code_llama)
    patch_data, label_data, url_data = vulfixminer_finetune.get_data(dataset_name)
    
    # 只处理测试集
    test_ids = []
    id_to_embeddings, id_to_label, id_to_url = {}, {}, {}
    index = 0
    
    print("Processing test data...")
    for i in tqdm(range(len(patch_data['test']))):
        label = label_data['test'][i]
        url = url_data['test'][i]
        embeddings = aggregator.transform(patch_data['test'][i])
        test_ids.append(index)
        id_to_embeddings[index] = embeddings
        id_to_label[index] = label
        id_to_url[index] = url
        
        # 统计 LOC
        total_loc = 0
        for diff in patch_data['test'][i]:
            added = vulfixminer_finetune.get_code_version(diff, added_version=True)
            deleted = vulfixminer_finetune.get_code_version(diff, added_version=False)
            total_loc += added.count('\n') + deleted.count('\n')
        loc_list.append(total_loc)
        index += 1

    test_set = VulFixMinerDataset(test_ids, id_to_label, id_to_embeddings, id_to_url)
    test_params = {
        'batch_size': TEST_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0,
        'collate_fn': custom_collate_fn
    }
    test_generator = DataLoader(test_set, **test_params)

    # 评估
    print("Evaluating on test set...")
    precision, recall, f1, auc, mcc, y_test, probs, y_pred = predict_test_data(
        model=model,
        testing_generator=test_generator,
        device=device,
        need_prob=True,
        prob_path=TEST_PROB_PATH
    )

    # 保存评估指标
    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mcc": mcc
    }
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")

    fieldnames = ["precision", "recall", "f1", "auc", "mcc"]
    
    if "java" in dataset_name.lower() or "python" in dataset_name.lower():
        print("\ncal Cost-Effectiveness ...")
        ce5 = cost_effort_at_L(y_test, probs, loc_list, L=5)
        ce10 = cost_effort_at_L(y_test, probs, loc_list, L=10)
        ce15 = cost_effort_at_L(y_test, probs, loc_list, L=15)
        ce20 = cost_effort_at_L(y_test, probs, loc_list, L=20)
        popt_5 = popt_at_L(y_test, probs, loc_list, L=5)
        popt_20 = popt_at_L(y_test, probs, loc_list, L=20)
        
        metrics_dict.update({
            "ce5": ce5,
            "ce10": ce10,
            "ce15": ce15,
            "ce20": ce20,
            "popt5": popt_5,
            "popt20": popt_20
        })
        fieldnames.extend(["ce5", "ce10", "ce15", "ce20", "popt5", "popt20"])
        
        print(f"CostEffort@5%: {ce5:.2f}%")
        print(f"CostEffort@10%: {ce10:.2f}%")
        print(f"CostEffort@15%: {ce15:.2f}%")
        print(f"CostEffort@20%: {ce20:.2f}%")
        print(f"Popt@5%: {popt_5:.4f}")
        print(f"Popt@20%: {popt_20:.4f}")
        
    elif "output1" in dataset_name.lower():
        print("\ncal FPR...")
        fpr = calculate_fpr(y_test, y_pred)
        metrics_dict["fpr"] = fpr
        fieldnames.append("fpr")
        print(f"False Positive Rate: {fpr:.4f}")

    # 保存评估结果到CSV
    metrics_path = MODEL_PATH.replace('.sav', '_evaluation_metrics.csv')
    with open(metrics_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics_dict)
    print(f"\nsaved: {metrics_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        # required=True,
                        # default='../Dataset/bigvul_VA_type_vulfixminer_dataset.csv',
                        # default='../Dataset/new_bigvul_VA_vulfixminer_dataset.csv',
                        # default='./Dataset/new_PatchDB_vulfixminer_dataset_ces.csv',
                        #default='Dataset/python.csv',
                        #default='Dataset/python.csv',
                        default='Dataset/java.csv',
                        help='name of dataset')
    parser.add_argument('--model_path',
                        type=str,
                        # required=True,
                        # default='model/new_bigvul_va_vulfixminer.sav',
                        # default='model/bigvul_va_type_vulfixminer.sav',
                        default='model/codellma_java1.sav',
                        #default='model/new_PatchDB_patch_vulfixminer_commitBART.sav',
                        #default='model/pyhton_vulfixminer_commitBART.sav',
                        #default='model/java_vulfixminer_commitBART.sav',
                        #default='model/java_patch_vulfixminer_codegen.sav',
                        #default='model/python_patch_vulfixminer_codegen.sav',
                        help='save train model to path')

    parser.add_argument('--finetune_model_path',
                        type=str,
                        # required=True,
                        # default='model/new_bigvul_va_vulfixminer_finetuned_model.sav',
                        # default='model/bigvul_va_type_vulfixminer_finetuned_model.sav',
                        # default='model/new_PatchDB_patch_vulfixminer_finetuned_model_ces.sav',
                        default='model/vulfixminer_MODEL.sav',
                        #default='model/commitbart_patchdb_patchrec_vulfixminer.sav',
                        #default='model/codegen25_7b_javaApyhton_vulfixminer.sav',
                        #default='model/commitbart_pyhton_java_vulfixminer.sav',
                        help='path to finetune file transfomer')

    parser.add_argument('--train_prob_path',
                        type=str,
                        # required=True,
                        #default='probs/new_patchDB_patchreco_vulfixminer_commitbart_train_probpy.txt',
                        #default='probs/java_patchreco_vulfixminer_codegen_train_prob.txt',
                        #default='probs/new_patchDB_patchreco_vulfixminer_train_probpy.txt',
                        # default='probs/new_patchDB_patch_vulfixminer_train_prob.txt',
                        # default='probs/new_VD_vulfixminer_train_prob.txt',
                        # default='probs/new_patchDB_patch_vulfixminer_train_prob.txt',
                        default='probs/codellama_java_vulfixminer_train_prob.txt',
                        help='')

    parser.add_argument('--test_prob_path',
                        type=str,
                        # required=True,
                        #default='probs/new_patchDB_patchreco_vulfixminer_test_probpy.txt',
                        #default='probs/new_patchDB_patchreco_vulfixminer_commitbart_test_probpy.txt',
                        # default='probs/new_patchDB_patch_vulfixminer_test_prob.txt',
                        #default='probs/java_patchreco_vulfixminer_codegen_test_prob.txt',
                        default='probs/codellama_java_vulfixminer_test_prob.txt',
                        help='')
   
    args = parser.parse_args()


    do_train(args)
