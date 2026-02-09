import ast
import json
import re
import statistics
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from rouge import Rouge

from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
# from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import os
import argparse
from collections import defaultdict

# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize



def preprocess_text(text,split=False):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    if split:
        return tokens
    else:
        return text

def metrics_bleu(result):
    references = [[preprocess_text(item["answer_gt"],split=True)] for item in result]
    hypotheses = [preprocess_text(item["answer"],split=True) for item in result]

    metrics = {}
    metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return metrics

def metrics_cider(result):
    references = {i: [item["answer_gt"]] for i, item in enumerate(result)}
    hypotheses = {i: [item["answer"]] for i, item in enumerate(result)}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)
    return {"CIDEr": cider_score}

def metrics_rouge(result):
    references = [item["answer_gt"] for item in result]
    hypotheses = [item["answer"] for item in result]
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(hypotheses, references, avg=True)
    return {"ROUGE-L": rouge_scores['rouge-l']['f']}

def metrics_meteor(result):
    references_meteor = {}
    hypotheses_meteor = {}
    
    for idx, entry in enumerate(result):
        qid = idx
        references_meteor[qid] = [entry['answer_gt']]
        hypotheses_meteor[qid] = [entry['answer']]
    # print(hypotheses_meteor)
    meteor_scorer = Meteor()
    meteor_score, score_per_instance = meteor_scorer.compute_score(references_meteor, hypotheses_meteor)
    return {"METEOR": meteor_score} 


def calc_metric_for_sentence_answer(result):
    """
    calculate BLEU4, CIDER, METEOR, ROUGE-L for sentence answers (temporal reasoning, perception reasoning)
    Returns:
        _type_: BLEU4,CIDER,METEOR,ROUGE-L
    """
    metrics_sentence={}

    bleu=metrics_bleu(result)
    metrics_sentence.update(bleu)

    cider=metrics_cider(result)
    metrics_sentence.update(cider)

    meteor=metrics_meteor(result)
    metrics_sentence.update(meteor)

    rouge=metrics_rouge(result)
    metrics_sentence.update(rouge)
    
    return metrics_sentence


def calculate_metric_for_temporal_reasoning(result):
    result_metrics_temporal=calc_metric_for_sentence_answer(result)
    return result_metrics_temporal

def calculate_metric_for_perception_reasoning(result):
    result_metrics_perception=calc_metric_for_sentence_answer(result)
    return result_metrics_perception


def calculate_metrics_for_description(data):
    """
    Args:
        data: A list of dictionaries containing score information
    
    Returns:
        dict: The average scores for each metric
    """
    score_counts = defaultdict(int)
    score_sums = defaultdict(float)
    metrics = set()
    
    for item in data:
        if "score" in item and isinstance(item["score"], dict):
            score_data = item["score"]   
           
            for metric, value in score_data.items():
                try:
                    numeric_value = float(value)
                    score_sums[metric] += numeric_value
                    score_counts[metric] += 1
                    metrics.add(metric)
                except (ValueError, TypeError):
                    
                    print(f"skipping invalid score: {value} for metric: {metric}")
                    continue
    
    # 计算平均值
    averages = {}
    for metric in metrics:
        if score_counts[metric] > 0:
            averages[metric] = score_sums[metric] / score_counts[metric]
    
    return averages

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Calculate metrics for model results.')
    parser.add_argument('--model_test_result_folder', type=str, required=True, help='Path to the model test result folder')
    args = parser.parse_args()
    model_test_result_folder = args.model_test_result_folder
    # model_test_result_folder ="/mnt/data2/wgk/surgvidlm_project/surgvidlm_open/test_result"
    result_path= os.path.join(model_test_result_folder, "clip_result.json")
    description_result_path=os.path.join(model_test_result_folder,"clip_result_description_scored.json")
    
    with open(result_path, 'r') as f:
        result = json.load(f)

    with open(description_result_path, 'r') as f:
        description_result = json.load(f)
    
    clip_description=[item for item in description_result if item['tag']=="video clip description"] 
    temporal=[item for item in result if item['tag']=="Visual Temporal Reasoning"]
    perception=[item for item in result if item['tag']=="Visual Perception Reasoning"]
    
    print(len(clip_description),len(temporal),len(perception))
    result_metrics={} #dict that store all metrics
    
    print("start calculating metrics: Visual Temporal Reasoning")
    result_metrics['visual_temporal_reasoning']=calculate_metric_for_temporal_reasoning(temporal)
    
    print("start calculating metrics: Visual Perception Reasoning")
    result_metrics['visual_perception_reasoning']=calculate_metric_for_perception_reasoning(perception)
    
    print("start calculating metrics: clip description")
    result_metrics['clip_description']=calculate_metrics_for_description(clip_description)
    
    print("evaluation finished, final metrics:")
    print(json.dumps(result_metrics, indent=4))
  

