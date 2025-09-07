import pandas as pd

# 加载训练数据
df = pd.read_parquet('/data/yangzhenfei/DisCO/datasets/deepscaler/data/train.parquet')

# 输出前10条内容
print("数据集形状:", df.shape)
print("\n前10条数据 (JSON格式):")
print(df.head(10).to_json(orient='records', indent=2, force_ascii=False))

# 打印列名
print("\n列名:")
print(df.columns.tolist())

# 打印数据类型
print("\n数据类型:")
print(df.dtypes)
