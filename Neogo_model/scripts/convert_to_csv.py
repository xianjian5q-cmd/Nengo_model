import pickle
import csv
import os
import numpy as np

# 输入pkl文件路径和输出csv文件路径
pkl_filepath = r'C:\Users\xianj\Desktop\thesis\improved\data\epa_dimensions.pkl'  # 替换为你的pkl文件路径
csv_filepath = r'C:\Users\xianj\Desktop\thesis\improved\data\epa_dimensions.csv'  # 输出csv文件路径

# 读取pkl文件
with open(pkl_filepath, 'rb') as f:
    vocab = pickle.load(f)

# 将数据转换为适合csv的格式
csv_data = []
for word, epa in vocab.items():
    # 将numpy数组转换为列表
    if isinstance(epa, np.ndarray):
        epa = epa.tolist()
    csv_data.append([word] + epa)

# 写入csv文件
with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入标题行
    writer.writerow(['Word', 'E', 'P', 'A'])
    # 写入数据
    writer.writerows(csv_data)

print(f'CSV文件已保存到: {csv_filepath}')
print(f'共转换了 {len(csv_data)} 条记录')
