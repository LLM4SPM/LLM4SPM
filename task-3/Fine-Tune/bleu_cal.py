import code2nl_cross.bleu as bleu
import json
def save_bleu_to_result_file(pred_file, gold_file, result_file):
    # 读取预测结果
    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f]

    # 计算 BLEU
    goldMap, predictionMap = bleu.computeMaps(predictions, gold_file)
    bleu4 = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    
    print(f"BLEU-4: {bleu4}")
    
    # 读取原有的 result 文件内容
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
    except Exception:
        result_json = {}
    
    # 将 BLEU-4 添加到 result 文件内容
    result_json["BLEU-4"] = bleu4
    
    # 写回结果文件
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

# 指定文件路径
pred_file = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ codereviewer/test_0.output"
gold_file = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ codereviewer/test_0.gold"
result_file = "/home/linxw/yzu-rza/code-xyf/result/code_msg_ codereviewer/result_0.output"

save_bleu_to_result_file(pred_file, gold_file, result_file)
