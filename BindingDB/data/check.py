import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 读取CSV文件
df = pd.read_csv('./bindingdb_data_ec.csv')  # 请将 'your_file.csv' 替换为你的CSV文件路径

# 提取EC50列
ec50 = df['log10EC']

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(ec50, bins=60, color='orange', edgecolor="black", alpha=0.7)
plt.title('EC50 Histogram')
plt.xlabel('EC50')
plt.ylabel('Frequency')
# plt.grid(True)
plt.savefig('../fig/EC50_dist.svg')
# 显示图形
plt.show()