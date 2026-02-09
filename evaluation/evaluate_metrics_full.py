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
    description_result_path=os.path.join(model_test_result_folder,"full_result_description_scored.json")

    with open(description_result_path, 'r') as f:
        description_result = json.load(f)
    
    full_description=[item for item in description_result if item['tag']=="video clip description"] 
    
    print(len(full_description))
    
    result_metrics={} #dict that store all metrics
    print("start calculating metrics: full video description")
    result_metrics['full_video_description']=calculate_metrics_for_description(full_description)
    
    print("evaluation finished, final metrics:")
    print(json.dumps(result_metrics, indent=4))
   

