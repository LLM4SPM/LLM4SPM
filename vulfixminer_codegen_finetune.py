# -*- coding: UTF-8 -*-
import os
import tiktoken
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/linxw/yzu-rza/vulfixminer1/vulfixminer/Dataset"

enc = tiktoken.get_encoding("gpt2")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 注释掉，让Accelerate管理设备
import csv

from transformers import RobertaTokenizer, RobertaModel,AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from torch.optim import AdamW
from transformers import get_scheduler
# from patch_entities import VulFixMinerFileDataset
from model import VulFixMinerFineTuneClassifier,VulFixMinerFineTuneClassifier_CodeGen25_7b
from tqdm import tqdm
import pandas as pd
from utils import get_code_version
import config
import argparse
from accelerate import Accelerator  # 新增：导入Accelerate库


# dataset_name = './Dataset/new_PatchDB_vulfixminer_dataset_ces.csv'
#dataset_name = './Dataset/output1.csv'
dataset_name='./Dataset/ase_dataset_sept_19_2021.csv'
# dataset_name = './Dataset/new_bigvul_VA_vulfixminer_dataset.csv'
# dataset_name = '../Dataset/bigvul_VA_type_vulfixminer_dataset.csv'
# dataset_name = '../Dataset/new_VD_vulfixminer_dataset.csv'
# dataset_name = '../Dataset/new_codejit_vulfixminer_dataset.csv'
# FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'
# FINE_TUNED_MODEL_PATH = './model/new_PatchDB_patch_vulfixminer_finetuned_model_ces.sav'
#FINE_TUNED_MODEL_PATH = './model/codegen25_7b_patchdb_vulfixminer.sav'
FINE_TUNED_MODEL_PATH = './model/codegen25_7b_javaApyhton_vulfixminer.sav'
# FINE_TUNED_MODEL_PATH = 'model/new_bigvul_va_vulfixminer_finetuned_model.sav'
# FINE_TUNED_MODEL_PATH = 'model/bigvul_va_type_vulfixminer_finetuned_model.sav'
# FINE_TUNED_MODEL_PATH = 'model/new_VD_vulfixminer_finetuned_model.sav'

# dataset_name = None
# FINE_TUNED_MODEL_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

# rerun with 5 finetune epoch
# FINETUNE_EPOCH = 5
#
# LIMIT_FILE_COUNT = 5
#
# NUMBER_OF_EPOCHS = 5
# TRAIN_BATCH_SIZE = 4
# VALIDATION_BATCH_SIZE = 32
# TEST_BATCH_SIZE = 32
# EARLY_STOPPING_ROUND = 5
# FINETUNE_EPOCH = 10
FINETUNE_EPOCH = 5
LIMIT_FILE_COUNT = 5
NUMBER_OF_EPOCHS = 5
# NUMBER_OF_EPOCHS = 10
TRAIN_BATCH_SIZE = 2
VALIDATION_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EARLY_STOPPING_ROUND = 5

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

LEARNING_RATE = 1e-5
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 256
HIDDEN_DIM = 4096
# HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.1
# HIDDEN_DIM_DROPOUT_PROB = 0.1
NUMBER_OF_LABELS = 2

class VulFixMinerFileDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_url, id_to_input, id_to_mask):
        self.list_IDs = list_IDs
        self.labels = labels
        self.id_to_url = id_to_url
        self.id_to_input = id_to_input
        self.id_to_mask = id_to_mask

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        url = self.id_to_url[id]
        input_id = self.id_to_input[id]
        mask = self.id_to_mask[id]

        y = self.labels[id]

        return int(id), url, input_id, mask, y

def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'][0], inputs.data['attention_mask'][0]


def predict_test_data(model, testing_generator, device, need_prob=False):
    # 初始化Accelerator以获取是否为主进程
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print("Testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    with torch.no_grad():
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(testing_generator, disable=not is_main_process):
            # 不需要手动移动到设备，Accelerator已经处理好了
            outs = model(input_batch, mask_batch)
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test, average='macro')
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test, average='macro')
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test, average='macro')

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    if is_main_process:
        print("Finish testing")
    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
    # 初始化Accelerator（会自动处理分布式环境）
    accelerator = Accelerator()
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"使用设备: {device}")
        print(f"进程数量: {accelerator.num_processes}")
        print(f"分布式类型: {accelerator.distributed_type}")
        print(f"混合精度: {accelerator.mixed_precision}")
    
    # 准备loss函数和优化器
    loss_function = nn.NLLLoss()
    
    # 使用PyTorch原生的AdamW优化器
    import torch.optim as optim
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # 这是关键步骤：使用accelerator准备所有组件，处理分布式训练
    model, optimizer, training_generator, test_generator = accelerator.prepare(
        model, optimizer, training_generator, test_generator
    )
    
    # 增加梯度累积以提高内存效率
    gradient_accumulation_steps = 4  # 根据需要调整
    
    train_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(training_generator, disable=not is_main_process):
            # accelerator已经处理好设备分配，无需手动移动张量
            outs = model(input_batch, mask_batch)
            outs = F.log_softmax(outs, dim=1)
            outs = outs.float()
            loss = loss_function(outs, label_batch.long())
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            train_losses.append(loss.item() * gradient_accumulation_steps)  # 记录实际损失
            
            # 使用accelerator处理反向传播
            accelerator.backward(loss)
            
            # 每gradient_accumulation_steps步更新一次
            if (current_batch + 1) % gradient_accumulation_steps == 0 or current_batch == len(training_generator) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.detach().item() * gradient_accumulation_steps
            current_batch += 1
            
            if current_batch % 50 == 0 and is_main_process:
                print(f"Train commit iter {current_batch}, total loss {np.sum(train_losses)}, average loss {np.average(train_losses)}")

        # 每个epoch结束后的处理
        if is_main_process:
            print(f"epoch {epoch}, training commit loss {np.sum(train_losses)}")
        train_losses = []

        model.eval()

        if is_main_process:
            print("Result on testing dataset...")
            
        precision, recall, f1, auc = predict_test_data(model=model,
                                                      testing_generator=test_generator,
                                                      device=device)
        
        if is_main_process:
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1: {f1}")
            print(f"AUC: {auc}")
            print("-" * 32)

        if epoch + 1 == FINETUNE_EPOCH:
            # 保存模型前先解包装
            unwrapped_model = accelerator.unwrap_model(model)
            if is_main_process:
                torch.save(unwrapped_model.state_dict(), FINE_TUNED_MODEL_PATH)
            
            # 冻结模型
            if hasattr(unwrapped_model, 'freeze_codebert'):
                unwrapped_model.freeze_codebert()
                
    return model

def retrieve_patch_data(all_data, all_label, all_url):
    # 初始化Accelerator以获取是否为主进程
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    tokenizer = AutoTokenizer.from_pretrained(
        "./CodeGen25_7b_mono",
        padding_side="left",
        trust_remote_code=True,
        use_fast=False
    )

    # ʽ
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token

    # ·ִ
    tokenizer.add_special_tokens({
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token
    })

    if is_main_process:
        print("Preparing tokenizer data...")
        print(f"pad_token: {tokenizer.pad_token}, unk_token: {tokenizer.unk_token}")

    id_to_label = {}
    id_to_url = {}
    id_to_input = {}
    id_to_mask = {}

    for i, diff in tqdm(enumerate(all_data), disable=not is_main_process):
        added_code = get_code_version(diff=diff, added_version=True)
        deleted_code = get_code_version(diff=diff, added_version=False)
        code = f"""
           ### Generate code patch analysis.
               Added code: {added_code}
               Deleted code: {deleted_code}
             """
        input_ids, mask = get_input_and_mask(tokenizer, code)
    
        id_to_input[i] = input_ids
        id_to_mask[i] = mask
        id_to_label[i] = all_label[i]
        id_to_url[i] = all_url[i]

    return id_to_input, id_to_mask, id_to_label, id_to_url


def get_sap_data(dataset_name):
    print(dataset_name)
    # 初始化Accelerator以获取是否为主进程
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}
    for item in items:
        try:
            commit_id = item[0]
            repo = item[1]
            url = repo + '/commit/' + commit_id
            partition = item[2]
            diff = item[3]
            label = item[4]
            pl = item[5]
            if url not in url_to_diff:
                url_to_diff[url] = []
            url_to_diff[url].append(diff)
            url_to_partition[url] = partition
            url_to_label[url] = label
            url_to_pl[url] = pl
        except (IndexError, TypeError) as e:
            #print(f"Error processing item {item}: {e}")
            continue
        except Exception as e:
            #print(f"Unexpected error processing item {item}: {e}")
            continue
    return url_to_diff, url_to_partition, url_to_label, url_to_pl


def get_tensor_flow_data(dataset_name):
    # 初始化Accelerator以获取是否为主进程
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[6]
        diff = item[4]

        if pd.isnull(diff):   
            continue
        
        label = item[5]
        pl = "UNKNOWN"

        if url not in url_to_diff:
            url_to_diff[url] = []

        url_to_diff[url].append(diff)
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    return url_to_diff, url_to_partition, url_to_label, url_to_pl


def get_data(dataset_name):
    # 初始化Accelerator以获取是否为主进程
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    # if dataset_name != 'sap_patch_dataset.csv':
    if dataset_name != 'sap_patch_dataset.csv':
        url_to_diff, url_to_partition, url_to_label, url_to_pl = get_sap_data(dataset_name)
    else:
        url_to_diff, url_to_partition, url_to_label, url_to_pl = get_tensor_flow_data(dataset_name) 

    patch_train, patch_test = [], []
    label_train, label_test = [], []
    url_train, url_test = [], []

    # with open('../Dataset/patchDB_time.csv', 'r') as file:
    # with open('../Dataset/va_type_time.csv', 'r') as file:
    # with open('../Dataset/va_time.csv', 'r') as file:
    # # with open('patchDB_cross.csv', 'r') as file:
    #     csv_reader = csv.reader(file)
    #     train_ids = next(csv_reader)
    #     test_ids = next(csv_reader)

    if is_main_process:
        print(len(url_to_diff.keys()))
    # diff here is diff list
    for key in url_to_diff.keys():
        url = key
        diff = url_to_diff[key]
        label = url_to_label[key]
        
        partition = url_to_partition[key]
        pl = url_to_pl[key]
        if partition == 'train':
            patch_train.append(diff)
            label_train.append(label)
            url_train.append(url)
        elif partition == 'test':
            patch_test.append(diff)
            label_test.append(label)
            url_test.append(url)

        # commit_id = key.split('/')[-1]
        # if commit_id in train_ids:
        #     # if len(patch_train) > 10:
        #     #     continue
        #     patch_train.append(diff)
        #     label_train.append(label)
        #     url_train.append(url)
        # elif commit_id in test_ids:
        #     # if len(patch_test) > 10:
        #     #     break
        #     patch_test.append(diff)
        #     label_test.append(label)
        #     url_test.append(url)

    if is_main_process:
        print("Finish reading dataset")
    patch_data = {'train': patch_train, 'test': patch_test}

    label_data = {'train': label_train, 'test': label_test}

    url_data = {'train': url_train, 'test': url_test}

    return patch_data, label_data, url_data


def do_train(args=None):
    global dataset_name, FINE_TUNED_MODEL_PATH

    # 初始化Accelerator
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process

    # dataset_name = args.dataset_path
    #
    # FINE_TUNED_MODEL_PATH = args.finetune_model_path

    if is_main_process:
        print("Dataset name: {}".format(dataset_name))
        print("Saving model to: {}".format(FINE_TUNED_MODEL_PATH))

    patch_data, label_data, url_data = get_data(dataset_name)

    train_ids, test_ids = [], []

    index = 0
    all_data, all_label, all_url = [], [], []

    for i in range(len(patch_data['train'])):
        label = label_data['train'][i]
        url = url_data['train'][i]
        for j in range(len(patch_data['train'][i])) :
            diff = patch_data['train'][i][j]
            train_ids.append(index)
            all_data.append(diff)
            all_label.append(label)
            all_url.append(url)
            index += 1

    for i in range(len(patch_data['test'])):
        label = label_data['test'][i]
        url = url_data['test'][i]
        for j in range(len(patch_data['test'][i])) :
            diff = patch_data['test'][i][j]
            test_ids.append(index)
            all_data.append(diff)
            all_label.append(label)
            all_url.append(url)
            index += 1


    if is_main_process:
        print("Preparing commit patch data...")
    id_to_input, id_to_mask, id_to_label, id_to_url = retrieve_patch_data(all_data, all_label, all_url)
    if is_main_process:
        print("Finish preparing commit patch data")
    
    training_set = VulFixMinerFileDataset(train_ids, id_to_label, id_to_url, id_to_input, id_to_mask)
    test_set = VulFixMinerFileDataset(test_ids, id_to_label, id_to_url, id_to_input, id_to_mask)
    train_loader_params = {**TRAIN_PARAMS, 'shuffle': False}  # 关闭shuffle
    training_generator = DataLoader(training_set, **train_loader_params)
    test_generator = DataLoader(test_set, **TEST_PARAMS)
    model = VulFixMinerFineTuneClassifier_CodeGen25_7b()
    
    # 不再需要手动移动模型到设备，Accelerator会处理
    # model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_generator=test_generator)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--dataset_path',
    #                     default='sap_patch_dataset.csv',
    #                     type=str,
    #                     required=True,
    #                     help='name of dataset')
    # parser.add_argument('--finetune_model_path',
    #                     default='finetune_model_path model/sap_patch_vulfixminer_finetuned_model.sav',
    #                     type=str,
    #                     required=True,
    #                     help='select path to save model')
    #
    # args = parser.parse_args()

    # do_train(args)
    
    do_train()
