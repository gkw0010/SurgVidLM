import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 本地与对比文件路径
FILE1 = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/surgvidlm/lora/video_clip_stage2_2fps_change_prompt_final/checkpoint-923/trainer_state.json"
FILE2 = "/mnt/data2/wgk/surgvidlm_project/baseline/LLaMA-Factory-0.9.2/saves/lora/qwen2-vl-7b-clip-ablation_wo_stage2/checkpoint-1423/trainer_state.json"

OUT = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/logs/loss_compare_sldwindow.png"
DPI = 220
XKEY = "step"     # 或 "epoch"
YKEY = "loss"
WINDOW = 41       # 滑动平均窗口（建议奇数，越大越平滑）
MIN_STEP = 400   # 只对比 step >= 400 的数据

def moving_avg(x, window=41):
    """居中滑动平均。边界采用扩展填充；数据少于窗口时自动缩小窗口。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    window = int(max(1, window))
    if window == 1 or len(x) < 3:
        return x
    if window % 2 == 0:  # 确保为奇数，方便居中
        window += 1
    if len(x) < window:
        window = len(x) if len(x) % 2 == 1 else len(x) - 1
        window = max(1, window)
        if window == 1:
            return x
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    y = np.convolve(xpad, kernel, mode="valid")
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
sm1 = moving_avg(l1, WINDOW) if len(l1) > 0 else np.array([])
sm2 = moving_avg(l2, WINDOW) if len(l2) > 0 else np.array([])

plt.figure(figsize=(8,5))
name1 = Path(FILE1).parent.name or "run1"
name2 = Path(FILE2).parent.name or "run2"

if len(sm1) > 0:
    plt.plot(s1, sm1, color="#0a4fa3", linewidth=2.0, alpha=0.95, label=f"{name1} (MA w={WINDOW})")
if len(sm2) > 0:
    plt.plot(s2, sm2, color="#d97706", linewidth=2.0, alpha=0.95, label=f"{name2} (MA w={WINDOW})")

plt.xlabel(XKEY)
plt.ylabel(YKEY)
plt.title(f"training loss ≥ {MIN_STEP} steps (moving average) comparison")
plt.legend()
plt.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"Saved figure to {OUT}")