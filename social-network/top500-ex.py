import json

# 定义输入和输出文件路径
input_file = r'F:\test_API\PACK\closeness_final_new.json'
output_file = r'output_top_200_closeness_centrality.json'

# 读取原始JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 确保数据是字典形式，并提取前500条
top_500_data = dict(list(data.items())[:500])

# 将前500条数据写入新的JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(top_500_data, f, ensure_ascii=False, indent=2)

print(f"前500条数据已保存到 {output_file}")
