import os
import re

# 配置路径
ROOT = "./augmentation/data/unc/train"
PREFIX = "unc_train_augtext_"  # 你的文件前缀
EXT = ".json"

# 获取所有目标文件
all_files = os.listdir(ROOT)
json_files = [f for f in all_files if f.endswith(EXT)]

# 提取序号的函数
def extract_number(filename):
    match = re.search(r'_(\d+)\.json$', filename)
    return int(match.group(1)) if match else -1

# 排序：按原序号升序
json_files_sorted = sorted(json_files, key=extract_number)

# 过滤掉无效文件（没有匹配到数字的）
valid_files = [f for f in json_files_sorted if extract_number(f) != -1]

print(f"共找到 {len(valid_files)} 个有效 JSON 文件，开始重命名...\n")

# 重命名
import tqdm
for new_idx, filename in tqdm.tqdm(enumerate(valid_files), total=len(valid_files), desc="重命名进度"):
    old_path = os.path.join(ROOT, filename)
    
    new_filename = f"{PREFIX}{new_idx}{EXT}"
    new_path = os.path.join(ROOT, new_filename)
    
    # 防止覆盖已存在的文件
    if os.path.exists(new_path):
        print(f"⚠️ 跳过：{new_path} 已存在！")
        continue
    
    os.rename(old_path, new_path)
    print(f"✅ 重命名: {filename} → {new_filename}")

print(f"\n✅ 重命名完成！总共处理 {len(valid_files)} 个文件。")