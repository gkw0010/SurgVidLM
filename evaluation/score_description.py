from gpt_eval_utils import *
import json
import os
import shutil
import argparse



def is_valid_number(s):
    return s is not None and len(s) == 1 and s.isdigit() and 1 <= int(s) <= 5


def get_description_metrics(calc_func,item):
     score=None
     cnt = 0
     print(calc_func)
     while (cnt < 3 and not is_valid_number(score)):

          if cnt >= 1:
               print("format is invalid")
          print(f"trying: {cnt} times")
          score = calc_func(item['question'], item['answer'], item['answer_gt'])
          cnt += 1
     return score

def score_description(result_path):
     scored_path = result_path.replace(".json", "description_scored.json") #check
     if not os.path.exists(scored_path):
          shutil.copy2(result_path, scored_path)
          print(f"file copied: {scored_path}")
     else:
          print(f"file already exists: {scored_path}")


     with open(scored_path, 'r', encoding='utf-8') as f:
          scored_result = json.load(f)

     # #check
     # scored_result=scored_result[1250:] #前1100个
     # # scored_result=scored_result[1100:1250] #前1100个
     print("scored_result:",len(scored_result), scored_result[0]['id'],scored_result[-1]['id'])

     scored_result_description=[item for item in scored_result if item['tag'] in ['full video detailed description','video clip description']]
     print(len(scored_result_description))

     for i,item in enumerate(scored_result_description):
          if "score" not in item:
               print(f"--------------------------------------Video {i}------------------------------")
               # print(item)
               correctness_score=""
               cnt=0

               correctness_score = get_description_metrics(get_correctness,item)
               detail_orientation_score = get_description_metrics(get_detail_orientation,item)
               context_score = get_description_metrics(get_context,item)
               temporal_score = get_description_metrics(get_temporal,item)

               scores={
                    "correctness":correctness_score,
                    "detail_orientation":detail_orientation_score,
                    "contextual_understanding":context_score,
                    "temporal_understanding":temporal_score
               }
               
               item['score']=scores
               print(f"Video {i} scores:", scores)
               with open(scored_path, 'w', encoding='utf-8') as f:
                    json.dump(scored_result_description, f, indent=4, ensure_ascii=False)

     print("Evaluation completed. Scored results saved to:", scored_path)

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description='Evaluate detailed descriptions from results.')
     parser.add_argument('--result_path', type=str, required=True, help='Path to the result JSON file')
     args = parser.parse_args()
     result_path = args.result_path

     score_description(result_path)
