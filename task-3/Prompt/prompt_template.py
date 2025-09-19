import os
from openai import OpenAI
import openai
import pandas as pd
import requests
import time
import numpy as np
from IPython.display import HTML, display
import math
from sklearn.metrics import precision_recall_curve, auc
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

current_dir = os.getcwd()


def OneChat(question):
    openai.api_key = "your_openai_api_key_here"
    openai.base_url = 'your_openai_api_url_here'
    client = OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    def get_response(question):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                # model="deepseek-chat",
                # model="claude-sonnet-4-20250514",
                messages=[
                    {"role": "system", "content": "You are now playing the role of a software security expert."},
                    {"role": "user", "content": "You are given a piece of code change. Please generate a description "
                                                "for the vulnerability fixed by the patch. Ensure the relevancy, "
                                                "conciseness, and fluency of your response. "
                                                f"Here is the code change: {question}. You are only allowed to "
                                                f"response in one sentence. "},
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    prediction = get_response(question)
    print(prediction)

    return prediction


def calculate_rouge_l(reference, candidate):
    """
    计算 ROUGE-L 分数
    :param reference: 参考文本 (str)
    :param candidate: 候选文本 (str)
    :return: ROUGE-L 的 precision, recall 和 f1 分数
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL']


def calculate_bleu_4(reference, candidate):
    """
    计算 BLEU-4 分数
    :param reference: 参考文本 (str)
    :param candidate: 候选文本 (str)
    :return: BLEU-4 分数
    """
    # 将文本分割为单词列表
    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    # 使用平滑函数处理短句子
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], cand_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing)
    return score


if __name__ == '__main__':
    PatchPair_dataset = pd.read_csv('pair_patch.csv')
    print(PatchPair_dataset)
    shuffled_df = PatchPair_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(0.9 * len(PatchPair_dataset))
    testing_set = shuffled_df[split_index:]

    print(testing_set)

    for index, row in testing_set.iterrows():
        print(row.commit_id)
        response = OneChat(row['diff'])
        testing_set.at[index, 'prediction'] = response
    print(testing_set)
    testing_set.to_csv(os.path.join(current_dir, 'DS_Predictions_FS.csv'), index=False)

    df = pd.read_csv('DS_Predictions_CoT.csv')
    total_sample = len(df)
    print(df)
    bleu_4 = 0
    rouge_l = 0
    for row in df.itertuples():
        if row.prediction.startswith(('<!DOCTYPE html>', 'Request timed out.')):
            total_sample -= 1
            continue
        rouge_temp = calculate_rouge_l(row.desc, row.prediction)
        bleu_temp = calculate_bleu_4(row.desc, row.prediction)
        rouge_f1_temp = rouge_temp.fmeasure
        rouge_l += rouge_f1_temp
        bleu_4 += bleu_temp

    print(f"\nBLEU-4 Score: {bleu_4 / total_sample:.5f}")
    print(f"Rouge-L: {rouge_l / total_sample:.5f}")
