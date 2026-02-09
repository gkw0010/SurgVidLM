from datasets import load_from_disk

path = "/data2/wgk/SurgVidLM/LLaMA-Factory/saves/tokenizer_video_clip_stage2_8fps"
ds_dict = load_from_disk(path)

# 1) 查看基本信息
print(ds_dict)
print(ds_dict["train"].features)
print(len(ds_dict["train"]))

# 2) 取第 i 条样本（比如第 0 条）
i = 0
ex = ds_dict["train"][i]
print(ex)                # 打印完整字典
print(ex.keys())         # 查看有哪些字段

# # 3) 仅打印关键信息，避免把长序列全部打出来
# def brief(ex, max_len=20):
#     out = {}
#     for k, v in ex.items():
#         if isinstance(v, list):
#             out[k] = f"list(len={len(v)}): {v[:max_len]}"
#         elif hasattr(v, "shape"):
#             out[k] = f"array(shape={v.shape}, dtype={getattr(v, 'dtype', None)})"
#         elif isinstance(v, bytes):
#             out[k] = f"bytes(len={len(v)})"
#         else:
#             out[k] = v
#     return out

# print(brief(ex))

# # 4) 随机抽样查看几条
# import random
# for _ in range(3):
#     j = random.randrange(len(ds_dict["train"]))
#     print(j, brief(ds_dict["train"][j]))

# # 5) 只取部分列查看
# cols = ["input_ids", "labels", "timecodes"]
# ex_partial = ds_dict["train"].select_columns(cols)[i]
# print(brief(ex_partial))