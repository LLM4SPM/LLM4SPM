import argparse
import json
import logging
import os
from collections import OrderedDict
import glob

# BLEU-4
import code2nl_cross.bleu as bleu
# ROUGE
from datasets import load_metric

from sentence_transformers import SentenceTransformer

# ======================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_idx_tsv(path):
    data = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                idx = str(len(data))
                text = line
            else:
                idx, text = parts[0].strip(), parts[1].strip()
            data[idx] = text
    return data


def process_files(gold_path, output_path, sbert_model):
    gold_map = load_idx_tsv(gold_path)
    pred_map = load_idx_tsv(output_path)
    common_ids = sorted(set(gold_map.keys()) & set(pred_map.keys()), key=lambda x: (len(x), x))
    if not common_ids:
        raise ValueError(f"No overlapping indices between {gold_path} and {output_path}!")

    refs = [gold_map[i] for i in common_ids]
    preds = [pred_map[i] for i in common_ids]

    # ===== BLEU-4 =====
    predictions_for_bleu = [f"{idx}\t{pred}" for idx, pred in zip(common_ids, preds)]
    goldMap, predictionMap = bleu.computeMaps(predictions_for_bleu, gold_path)
    bleu4 = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 5)
    logger.info(f"BLEU-4: {bleu4}")

    # ===== ROUGE-L =====
    metric_rouge = load_metric("rouge", trust_remote_code=True)
    rouge_result = metric_rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    rougeL = round(rouge_result["rougeL"].mid.fmeasure * 100, 5)
    logger.info(f"ROUGE-L: {rougeL}")

    # ===== Sentence-BERT =====
    sbert = SentenceTransformer(sbert_model)
    emb_pred = sbert.encode(preds, convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = sbert.encode(refs, convert_to_tensor=True, normalize_embeddings=True)

    if hasattr(sbert, "similarity"):
        sim_mat = sbert.similarity(emb_pred, emb_ref)  # [N, N] pairwise 相似度
    else:
        from sentence_transformers import util as st_util
        sim_mat = st_util.cos_sim(emb_pred, emb_ref)

    pair_cos = sim_mat.diag().cpu().tolist()  
    sbert_avg = float(sum(pair_cos) / len(pair_cos))  
    logger.info(f"SentenceBERT avg cosine: {sbert_avg:.6f}")

    return bleu4, rougeL, sbert_avg


def scan_and_process(directory, sbert_model, save_summary_json=None):
    summary_results = []
    for subdir, _, _ in os.walk(directory):
        gold_file = os.path.join(subdir, "test_0.gold")
        output_file = os.path.join(subdir, "test_0.output")
        logger.info(f"Checking directory: {subdir}")
        logger.info(f"Looking for {gold_file} and {output_file}")

        if os.path.exists(gold_file) and os.path.exists(output_file):
            logger.info(f"Processing: {subdir}")
            try:
                bleu4, rougeL, sbert_avg = process_files(gold_file, output_file, sbert_model)
                result = {
                    "directory": subdir,
                    "BLEU-4": bleu4,
                    "ROUGE-L": rougeL,
                    "SentenceBERT_avg_cosine": round(sbert_avg, 6),
                    "gold_file": os.path.abspath(gold_file),
                    "pred_file": os.path.abspath(output_file),
                }
                summary_results.append(result)
            except ValueError as e:
                logger.warning(f"Skipping {subdir} due to error: {str(e)}")
    if save_summary_json:
        with open(save_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to: {save_summary_json}")

    print(json.dumps(summary_results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE-L, BLEU-4 from .gold and .output")
    parser.add_argument("--directory", default='.', help="要扫描的目录，查找其中的子目录和文件")
    parser.add_argument("--save_summary_json", default=None, help="保存汇总结果的 JSON 文件路径")
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2", help="使用的 Sentence-BERT 模型名")
    args = parser.parse_args()
    scan_and_process(args.directory, args.sbert_model, args.save_summary_json)


if __name__ == "__main__":
    main()


