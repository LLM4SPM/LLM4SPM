import os
from openai import OpenAI
import openai
import pandas as pd
import requests
import time
import numpy as np
from IPython.display import HTML, display
import math
import json
from sklearn.metrics import precision_recall_curve, roc_auc_score
from datetime import datetime

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
                    {"role": "user", "content": "You are given a piece of code change. If it is a security patch, "
                                                "please answer 'yes', otherwise answer 'no'. Only reply with one of "
                                                "the options above. Do not include any further information.\n Here is "
                                                f"the code change: {question}. "},
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    prediction = get_response(question)
    print(prediction)

    return prediction

if __name__ == '__main__':
    """ PatchDB """
    dataset = pd.read_json('patch_db.json')

    shuffled_df = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(0.9 * len(dataset))
    testing_set = shuffled_df[split_index:]

    for index, row in testing_set.iterrows():
        print(row.commit_id)
        response, prob = OneChat(row.diff_code)
        testing_set.at[index, 'prediction'] = response
    print(testing_set)
    testing_set.to_csv(os.path.join(current_dir, 'prob_LLM_PatchDB_DS_Predictions_FS.csv'), index=False)


    # """ VulFix-Python """
    # VulFix_dataset = pd.read_csv('ase_dataset_sept_19_2021.csv')
    # py = VulFix_dataset[VulFix_dataset.PL == 'python']
    # java = VulFix_dataset[VulFix_dataset.PL == 'java']
    # py_test = py[py.partition == "test"]
    # java_test = java[java.partition == "test"]
    #
    # shuffled_df = py_test.sample(frac=1, random_state=42).reset_index(drop=True)
    # split_index = int(0.95 * len(py_test))
    # testing_set_python = shuffled_df[split_index:].drop_duplicates(subset='commit_id')
    #
    # print(testing_set_python.label.value_counts())
    # print(testing_set_python)
    #
    # testing_set_python['prediction'] = None
    # testing_set_python['prob'] = None
    #
    # # delay_seconds = 2 * 60 * 60
    # # print(f"等待 {delay_seconds} 秒（2 小时）后执行...")
    # # time.sleep(delay_seconds)
    #
    # for index, row in testing_set_python.iterrows():
    #     print(row.commit_id)
    #     response, prob = OneChat(row['diff'])
    #     testing_set_python.at[index, 'prediction'] = response
    #     testing_set_python.at[index, 'prob'] = prob
    # print(testing_set_python)
    # testing_set_python.to_csv(os.path.join(current_dir, 'VulFix_Python_DS_FS.csv'), index=False)
    #
    # """ VulFix-Java """
    # shuffled_df_Java = java_test.sample(frac=1, random_state=42).reset_index(drop=True)
    # split_index = int(0.85 * len(java_test))
    # testing_set_Java = shuffled_df_Java[split_index:].drop_duplicates(subset='commit_id')
    #
    # print(testing_set_Java.label.value_counts())
    # print(testing_set_Java)
    #
    # testing_set_Java['prediction'] = None
    # testing_set_Java['prob'] = None
    #
    # for index, row in testing_set_Java.iterrows():
    #     print(row.commit_id)
    #     response, prob = OneChat(row['diff'])
    #     testing_set_Java.at[index, 'prediction'] = response
    #     testing_set_Java.at[index, 'prob'] = prob
    # print(testing_set_Java)
    # testing_set_Java.to_csv(os.path.join(current_dir, 'VulFix_Java_DS_FS.csv'), index=False)
    #
    # print(e)

    df = pd.read_csv('diversevul_gpt.csv')
    # print(df)
    # print(e)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # label = []
    # prediction_prob = []
    # line_label_prob = []
    # total_loc = 0
    # threshold = 0.85
    for row in df.itertuples():
        if row.target == 0 and row.prediction == 'no':
            TN += 1
            # label.append(0)
            # prediction_prob.append(1-row.prob)
            # line_label_prob.append((row.label, 1-row.prob, row.LOC_MOD, row.prediction))
        if row.target == 0 and row.prediction == 'yes':
            FP += 1
            # label.append(0)
            # prediction_prob.append(row.prob)
            # line_label_prob.append((row.label, row.prob, row.LOC_MOD, row.prediction))
        if row.target == 1 and row.prediction == 'no':
            FN += 1
            # label.append(1)
            # prediction_prob.append(1-row.prob)
            # line_label_prob.append((row.label, 1-row.prob, row.LOC_MOD, row.prediction))
        if row.target == 1 and row.prediction == 'yes':
            TP += 1
        #     label.append(1)
        #     prediction_prob.append(row.prob)
        #     line_label_prob.append((row.label, row.prob, row.LOC_MOD, row.prediction))
        # total_loc += row.LOC_MOD

    print(TP)
    print(FP)
    print(TN)
    print(FN)
    print(TP + FP + TN + FN)
    # print(label)
    # print(prediction_prob)
    # print(line_label_prob)

    # sorted_by_prob = sorted(line_label_prob, key=lambda x: x[2], reverse=True)
    # check_loc = threshold*total_loc
    # # print(sorted_by_prob)
    # print(total_loc)
    # print(check_loc)
    # current_loc = 0
    # true_positive = 0
    # total_positive = TP+FN
    # print(total_positive)
    # for commit in sorted_by_prob:
    #     if current_loc + commit[2] < check_loc:
    #         current_loc += commit[2]
    #         if commit[0] == 1 and commit[3] == 'yes':
    #             true_positive += 1
    #     else:
    #         break
    #
    # print(true_positive)

    # print(e)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    FPR = FP / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    print('Accuracy: ' + str(Accuracy))
    print('Precision: ' + str(Precision))
    print('Recall: ' + str(Recall))
    print('F1: ' + str(F1))
    print('FPR: ' + str(FPR))
    print('MCC: ' + str(MCC))
