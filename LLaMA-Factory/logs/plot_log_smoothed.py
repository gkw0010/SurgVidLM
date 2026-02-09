import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 本地与对比文件路径
FILE1 = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/lora/video_clip_stage2_4fps_change_prompt/checkpoint-71/trainer_state.json"
FILE2 = "/mnt/data2/wgk/surgvidlm_project/baseline/LLaMA-Factory-0.9.2/saves/lora/qwen2-vl-7b-clip-ablation_wo_stage2/trainer_log.jsonl"

OUT = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/logs/loss_compare.png"
DPI = 350
XKEY = "step"     # 或 "epoch"
YKEY = "loss"
ALPHA = 0.05      # EMA 平滑系数
MIN_STEP = 0    # 只对比 step >= 400 的数据

def ema(x, alpha=0.08):
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def load_xy(file, xkey="step", ykey="loss", min_step=None):
    with open(file, "r", encoding="utf-8") as f:
        logs = json.load(f)["log_history"]
    xs, ys = [], []
    for r in logs:
        if xkey in r and ykey in r and r[xkey] is not None and r[ykey] is not None:
            x = float(r[xkey])
            y = float(r[ykey])
            if (min_step is None) or (x >= min_step):
                xs.append(x)
                ys.append(y)
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return xs, ys

# 读取并平滑（仅 step >= 400）
s1, l1 = load_xy(FILE1, XKEY, YKEY, min_step=MIN_STEP)
s2, l2 = load_xy(FILE2, XKEY, YKEY, min_step=MIN_STEP)
sm1 = ema(l1, ALPHA) if len(l1) > 0 else np.array([])
sm2 = ema(l2, ALPHA) if len(l2) > 0 else np.array([])

plt.figure(figsize=(8,5))
name1 = Path(FILE1).parent.name or "run1"
name2 = Path(FILE2).parent.name or "run2"

if len(sm1) > 0:
    plt.plot(s1, sm1, color="#0a4fa3", linewidth=2.0, alpha=0.95, label=f"stage 2(EMA α={ALPHA})")
if len(sm2) > 0:
    plt.plot(s2, sm2, color="#d97706", linewidth=2.0, alpha=0.95, label=f"wo stage2 (EMA α={ALPHA})")

plt.xlabel(XKEY)
plt.ylabel(YKEY)
plt.title(f"training loss ≥ {MIN_STEP} steps (smoothed) comparison")
plt.legend()
plt.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"Saved figure to {OUT}")