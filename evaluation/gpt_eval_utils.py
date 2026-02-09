import json

import requests


def get_correctness(question,answer_pred,answer_gt):
    '''
    get correctness of the answer
    '''
    # OpenAI API Key
    url = "https://api.zhiyungpt.com/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        #    'Authorization': 'sk-7kT91fwVRKEpGjAkUoxVNXZTIkXH3guWO09HXkarZEbOiZwQ',
        'Authorization': 'sk-kE97HyzqXtcyj0grzJy8xTIio8xa1AxQDirSzi4tzKBMTnx9',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'}

    # print("system input: ",total_prompt)
    payload = json.dumps({
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                    "- The predicted answer must be factually accurate and align with the video content.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the factual accuracy of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer_gt}\n"
                    f"Predicted Answer: {answer_pred}\n\n"
                    "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 1 and 5, with 5 indicating the highest level of factual accuracy. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the integer value."
                    "For example, your response should look like this: 1"
                # "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."

            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    })

    json_1 = None
    # print(payload.j ['messages'][1]['content'])

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        # print(response.json())
        json_1 = response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print("请求超时或回答被过滤")
        print(e)
    print(json_1)
    return json_1


def get_detail_orientation(question,answer_pred,answer_gt):
    '''
    获取detailed orientation
    '''
    #OpenAI API Key
    url = "https://api.zhiyungpt.com/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        #    'Authorization': 'sk-7kT91fwVRKEpGjAkUoxVNXZTIkXH3guWO09HXkarZEbOiZwQ',
        'Authorization': 'sk-kE97HyzqXtcyj0grzJy8xTIio8xa1AxQDirSzi4tzKBMTnx9',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'}

    # print("system input: ",total_prompt)
    payload = json.dumps({
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                    "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer_gt}\n"
                    f"Predicted Answer: {answer_pred}\n\n"
                    "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 1 and 5, with 5 indicating the highest level of detail orientation. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the integer value. "
                    "For example, your response should look like this: 1"
                # "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."

            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    })

    json_1 = None
    # print(payload.j ['messages'][1]['content'])

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        json_1 = response.json()["choices"][0]["message"]["content"]
    # print(response.json())
    except Exception as e:
        print("请求超时或回答被过滤")
        print(e)
    print(json_1)
    return json_1


def get_context(question, answer_pred, answer_gt):

    # OpenAI API Key
    url = "https://api.zhiyungpt.com/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        #    'Authorization': 'sk-7kT91fwVRKEpGjAkUoxVNXZTIkXH3guWO09HXkarZEbOiZwQ',
        'Authorization': 'sk-kE97HyzqXtcyj0grzJy8xTIio8xa1AxQDirSzi4tzKBMTnx9',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'}

    # print("system input: ",total_prompt)
    payload = json.dumps({
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                    "- The predicted answer must capture the main themes and sentiments of the video.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Provide your evaluation of the contextual understanding of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer_gt}\n"
                    f"Predicted Answer: {answer_pred}\n\n"
                    "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 1 and 5, with 5 indicating the highest level of contextual understanding. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the integer value. "
                    "For example, your response should look like this: 1"
                # "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."

            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    })

    json_1 = None
    # print(payload.j ['messages'][1]['content'])

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        json_1 = response.json()["choices"][0]["message"]["content"]
    # print(response.json())
    except Exception as e:
        print(e)
        print("请求超时或回答被过滤")
    print(json_1)
    return json_1


def get_temporal(question, answer_pred, answer_gt):

    # OpenAI API Key
    url = "https://api.zhiyungpt.com/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
           'Authorization': 'sk-kE97HyzqXtcyj0grzJy8xTIio8xa1AxQDirSzi4tzKBMTnx9',
        # 'Authorization': 'sk-FpPbaKbLN646DX8iIIcPolhlztsgDMEu6sDHeYyQP4SSjUW7',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'}

    # print("system input: ",total_prompt)
    payload = json.dumps({
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                    "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                    "- Evaluate the temporal accuracy of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer_gt}\n"
                    f"Predicted Answer: {answer_pred}\n\n"
                    "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 1 and 5, with 5 indicating the highest level of temporal consistency. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the number. "
                    "For example, your response should look like this: 1"
                # "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."

            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    })

    json_1 = None
    # print(payload.j ['messages'][1]['content'])

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        json_1 = response.json()["choices"][0]["message"]["content"]
        # print(response.json())
    # print(response.json())
    except Exception as e:
        print("请求超时或回答被过滤")
        print(e)
    print(json_1)
    return json_1

def get_consistency(question, answer_pred, answer_gt):

    # OpenAI API Key
    url = "https://api.zhiyungpt.com/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        #    'Authorization': 'sk-7kT91fwVRKEpGjAkUoxVNXZTIkXH3guWO09HXkarZEbOiZwQ',
        'Authorization': 'sk-kE97HyzqXtcyj0grzJy8xTIio8xa1AxQDirSzi4tzKBMTnx9',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'}

    # print("system input: ",total_prompt)
    payload = json.dumps({
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                    "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                    "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                    "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                    "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                    "- Evaluate the consistency of the two predicted answers compared to the correct answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer_gt}\n"
                    f"Predicted Answer: {answer_pred}\n\n"
                    "Provide your evaluation only as a consistency score where the consistency score is an numerical value between 1 and 5, rounded to one decimal place, with 5 indicating the highest level of consistency. "
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the number. "
                    "For example, your response should look like this: 4."
                # "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."

            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    })

    json_1 = None
    # print(payload.j ['messages'][1]['content'])

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        json_1 = response.json()["choices"][0]["message"]["content"]
    # print(response.json())
    except:
        print("请求超时或回答被过滤")
    print(json_1)
    return json_1