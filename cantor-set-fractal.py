## cantor-set-fractal.py
# PyTorch implementation of the Cantor set with vectorized tensor ops.

import torch
import matplotlib.pyplot as plt

# ---- 1) 选择计算设备（如果有 GPU 就用 GPU）------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- 2) 核心：用张量并行生成康托集的各层区间 -----------------------------
# intervals 的每一行是 [start, length]，初始为 [0, 1]
def build_cantor_intervals(n_iters: int):
    # 1x2 的起始区间张量（在 device 上）
    intervals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)

    levels = []   # 为了绘图，保存每一层的区间（放到 CPU 上）
    for _ in range(n_iters):
        # 保存这一层（绘图用）
        levels.append(intervals.detach().cpu())

        # --- 关键：一次性并行分裂所有区间，而不是 for 循环 ---
        s = intervals[:, 0]       # 所有区间的 start
        L = intervals[:, 1]       # 所有区间的 length
        third = L / 3.0

        # 左右子区间： [s, L/3] 与 [s+2/3 L, L/3]
        left  = torch.stack((s, third), dim=1)
        right = torch.stack((s + 2.0 * third, third), dim=1)

        # 新一层的全部区间 = 左 + 右（张量拼接，仍在 device 上）
        intervals = torch.cat((left, right), dim=0)

    # 末层也保存下来
    levels.append(intervals.detach().cpu())
    return levels

# ---- 3) 绘图：把每一层画成一条横线上的多个黑色小段 ------------------------
def plot_cantor(levels, line_width=6):
    plt.figure(figsize=(9, 5))
    depth = len(levels)

    # 从上到下画（第 0 层在最上方）
    for i, layer in enumerate(levels):
        y = depth - 1 - i          # 让第 0 层在最上面
        # layer 每行：[start, length]
        starts  = layer[:, 0].numpy()
        lengths = layer[:, 1].numpy()
        ends = starts + lengths

        # 用少量循环把每个小段画出来（绘图允许循环；计算已用并行）
        for x0, x1 in zip(starts, ends):
            plt.hlines(y, x0, x1, colors="black", linewidth=line_width)

    plt.xlim(0, 1)
    plt.ylim(-0.5, depth - 0.5)
    plt.yticks(range(depth), [f"Level {i}" for i in range(depth)])
    plt.xlabel("x")
    plt.title(f"Cantor Set (PyTorch, device={device.type})")
    plt.tight_layout()
    plt.show()

# ---- 4) 运行示例 ---------------------------------------------------------
if __name__ == "__main__":
    # 迭代层数：6~8 层就能得到与 Wikipedia 类似的条纹效果
    N_ITERS = 7
    levels = build_cantor_intervals(N_ITERS)
    print("Segments on last level:", len(levels[-1]))
    plot_cantor(levels)
