import numpy as np
import matplotlib.pyplot as plt

# 模拟数据
epochs = np.linspace(0, 10, 200)
log_prob_initial = -130 + 5 * np.log1p(epochs) / np.log1p(10)         # 黑线略有上升
log_prob_ground_truth = -140 - 10 * epochs + np.random.randn(200)    # 红线持续下降
log_prob_final = -145 + 7 * np.log1p(epochs) / np.log1p(10)           # 蓝线也上升，略高于黑线

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(epochs, log_prob_initial, color='black', linewidth=2, label='Initial Argmax')
plt.plot(epochs, log_prob_ground_truth, color='red', linewidth=2, label='Ground Truth $y_u$')
plt.plot(epochs, log_prob_final, color='blue', linewidth=2, label='Final Response $\~{y}_u$')

# 标签与图例
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Log-probability", fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.grid(False)

# 展示
plt.tight_layout()
plt.show()
