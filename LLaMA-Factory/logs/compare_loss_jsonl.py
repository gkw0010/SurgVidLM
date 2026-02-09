import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 文件路径
FILE1 = "/mnt/data2/wgk/LLaMA-Factory/saves/lora/qwen2_vl_full_new/trainer_log.jsonl"  # JSONL
FILE2 = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/surgvidlm/lora/video_clip_stage2_2fps_change_prompt_final/trainer_state.json"  # 原 JSON
NEW_FILE2 = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/surgvidlm/lora/video_clip_stage2_2fps_change_prompt_final/trainer_state_odd.json"  # 生成的新 JSON（奇数索引）

OUT = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/logs/loss_compare.png"
DPI = 350

# 你可以改为 "epoch"
XKEY = "step"     # 可选: "step" 或 "epoch"
YKEY = "loss"
ALPHA = 0.05      # EMA 平滑系数
MIN_STEP = 0      # 只对比 x >= MIN_STEP 的数据

def ema(x, alpha=0.08):
    if len(x) == 0:
        return np.array([])
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def build_odd_index_json(src_file, dst_file):
    """
    从 src_file(trainer_state.json) 读取，取 log_history 的奇数索引项 [1::2]，
    写回到 dst_file，保留 trainer_state 的其余顶层键（若存在）。
    """
    with open(src_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    logs = obj.get("log_history", [])
    odd_logs = logs[1::2]  # 取索引 1,3,5,...

    new_obj = dict(obj)
    new_obj["log_history"] = odd_logs

    Path(dst_file).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_file, "w", encoding="utf-8") as f:
        json.dump(new_obj, f, ensure_ascii=False)

    print(f"Built odd-index json: {dst_file} with {len(odd_logs)} records (from {len(logs)})")

def load_xy_trainer_state_json(file, xkey="step", ykey="loss", min_step=None):
    with open(file, "r", encoding="utf-8") as f:
        logs = json.load(f)["log_history"]

    print(f"Loaded {len(logs)} records from {file}")
    xs, ys = [], []
    for r in logs:
        if xkey in r and ykey in r and r[xkey] is not None and r[ykey] is not None:
            x = float(r[xkey]/2-1)
            y = float(r[ykey])
            if (min_step is None) or (x >= min_step):
                xs.append(x)
                ys.append(y)
    if len(xs) == 0:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return xs, ys

def load_xy_jsonl(file, xkey="step", ykey="loss", min_step=None):
    """
    读取 JSONL：每行形如
    {"current_steps": 176, "total_steps": 1423, "loss": 1.6836, "lr": ..., "epoch": 0.12, ...}
    - 当 xkey == "step" 时，用 current_steps
    - 当 xkey == "epoch" 时，用 epoch
    - ykey 默认为 "loss"
    """
    x_field = "current_steps" if xkey == "step" else xkey  # 支持 "epoch"
    y_field = ykey

    xs, ys = [], []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x_field in r and y_field in r and r[x_field] is not None and r[y_field] is not None:
                try:
                    x = float(r[x_field])
                    y = float(r[y_field])
                except (TypeError, ValueError):
                    continue
                if (min_step is None) or (x >= min_step):
                    xs.append(x)
                    ys.append(y)
    if len(xs) == 0:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return xs, ys

# 先构建只含奇数位置记录的新 JSON
build_odd_index_json(FILE2, NEW_FILE2)

# 读取并平滑
s1, l1 = load_xy_jsonl(FILE1, XKEY, YKEY, min_step=MIN_STEP)               # 从 JSONL 读
s2, l2 = load_xy_trainer_state_json(NEW_FILE2, XKEY, YKEY, min_step=MIN_STEP)  # 从新 JSON 读（奇数索引）
print(s2,l2)

sm1 = ema(l1, ALPHA) if len(l1) > 0 else np.array([])
sm2 = ema(l2, ALPHA) if len(l2) > 0 else np.array([])

plt.figure(figsize=(8, 5))
name1 = Path(FILE1).parent.name or "run1"
name2 = Path(FILE2).parent.name or "run2"

if len(sm2) > 0:
    plt.plot(s2, sm2, color="#d97706", linewidth=1.0, alpha=0.95,
             label=f"wo stage2 (EMA α={ALPHA})")

if len(sm1) > 0:
    plt.plot(s1, sm1, color="#0a4fa3", linewidth=1.0, alpha=0.95,
             label=f"stage 2 (EMA α={ALPHA})")

xlabel = XKEY if XKEY != "step" else "step"
plt.xlabel(xlabel)
plt.ylabel(YKEY)
plt.title(f"training {YKEY} ≥ {MIN_STEP} {xlabel}s (smoothed) comparison")
plt.legend()
plt.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"Saved figure to {OUT}")