import json
import numpy as np
import matplotlib.pyplot as plt

FILE = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/saves/surgvidlm/lora/video_clip_stage2_2fps_change_prompt_final/checkpoint-923/trainer_state.json"   # 修改为你的路径
OUT = "/mnt/data2/wgk/surgvidlm_project/LLaMA-Factory/logs/loss_curve.png"        # 输出文件名：.png/.svg/.pdf 等
DPI = 200                     # 分辨率
XKEY = "step"                 # 或 "epoch"
YKEY = "loss"
USE_EMA = True                # False 切换为滑动平均
ALPHA = 0.08                  # EMA 平滑系数
WINDOW = 41                   # 滑动平均窗口（奇数）

def ema(x, alpha=0.1):
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def moving_avg(x, window=41):
    if window <= 1 or len(x) < window:
        return x.astype(float)
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(window) / window
    return np.convolve(xpad, ker, mode="valid")

# 读取与提取
with open(FILE, "r", encoding="utf-8") as f:
    logs = json.load(f)["log_history"]

steps, losses = [], []
for r in logs:
    if XKEY in r and YKEY in r and r[XKEY] is not None and r[YKEY] is not None:
        steps.append(float(r[XKEY]))
        losses.append(float(r[YKEY]))

# 排序并转为数组
order = np.argsort(steps)
steps = np.array(steps)[order]
losses = np.array(losses)[order]

# 平滑
smooth = ema(losses, ALPHA) if USE_EMA else moving_avg(losses, WINDOW)
label = f"smoothed (EMA α={ALPHA})" if USE_EMA else f"smoothed (MA w={WINDOW})"

# 作图并保存
plt.figure(figsize=(8,5))
plt.plot(steps, losses, color="#86b6f6", alpha=0.45, linewidth=1.0, label="original")
plt.plot(steps, smooth, color="#0a4fa3", alpha=0.95, linewidth=2.0, label=label)
plt.xlabel(XKEY)
plt.ylabel(YKEY)
plt.title("training loss")
plt.legend()
plt.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"Saved figure to {OUT}")