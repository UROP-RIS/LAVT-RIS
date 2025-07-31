# file: async_maskris_pipeline.py
"""
高性能异步数据增强管线
- 多线程加载 .npz 文件
- 异步并发调用 DeepSeek API（async + httpx）
- 每批处理 8 条，最多 500 条
- 每条输出单独的 JSON 文件
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio
from json_repair import repair_json

from tqdm.asyncio import tqdm_asyncio


# ======================
# 配置区
# ======================

# DeepSeek API 配置
API_KEY = "sk-da41ed5334f5457ebfb96ff3aa2ea6e1"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# 数据与任务配置
DATASET_ROOT = "/data/datasets/tzhangbu/Cherry-Pick/data/refcoco"
DATASET = "unc"
SPLIT = "train"
MAX_ITEMS = None           # 只处理前 500 条
BATCH_SIZE = 10           # 每批并发数
MAX_WORKERS = 16          # 多线程加载文件
OUTPUT_DIR = f"./augmentation/data/{DATASET}/{SPLIT}"  # 输出目录（每个样本一个文件）
# 请求参数
TEMPERATURE = 0.75
TOP_P = 0.95
MAX_TOKENS = 2048
MAX_CONCURRENT = 100      # 最大并发请求数

TIMEOUT = 180.0
MAX_RETRIES = 3
AUG_NUM = 3

# ======================
# 同步：加载单个 .npz 文件，返回 (text, idx)
# ======================
def load_data(file_path: str) -> Dict[str, str]:
    try:
        # 从文件名中提取编号
        filename = os.path.basename(file_path)
        match = re.search(r'_(\d+)\.npz$', filename)
        idx = match.group(1) if match else "unknown"

        img_txt_gt = np.load(file_path, allow_pickle=True)
        data_dict = {key: img_txt_gt[key] for key in img_txt_gt}
        text = str(data_dict['sent_batch'][0]).strip()
        return {"text": text, "idx": idx}
    except Exception as e:
        print(f"❌ 加载失败 {file_path}: {e}")
        return {"text": "", "idx": "unknown"}


# ======================
# 同步：多线程加载所有数据（带编号）
# ======================
def load_all_data(data_root: str, max_items: int = None) -> List[Dict[str, str]]:
    files = [f for f in os.listdir(data_root) if f.endswith('.npz')]

    def extract_number(filename):
        match = re.search(r'_(\d+)\.npz$', filename)
        return int(match.group(1)) if match else -1

    files_sorted = sorted(files, key=extract_number)
    if max_items is not None:
        files_sorted = files_sorted[:max_items]
    file_paths = [os.path.join(data_root, f) for f in files_sorted]

    print(f"📄 正在加载前 {len(file_paths)} 个 .npz 文件...")
    data_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(load_data, fp) for fp in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="📂 加载数据"):
            result = future.result()
            if result["text"]:
                data_list.append(result)
            else:
                # 即使文本为空，也保留编号
                data_list.append(result)

    return data_list


# ======================
# 异步：调用 DeepSeek 批量增强（核心：正确使用 semaphore）
# ======================
async def async_batch_augment(
    client: httpx.AsyncClient,
    items: List[Dict[str, str]],  # 每个 item: {"text": "...", "idx": "123"}
    batch_id: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    aug_num=3,
) -> Optional[List[Dict]]:
    """
    在 semaphore 保护下执行请求，限制并发数
    """
    async with semaphore:
        texts = [item["text"] for item in items]
        input_section = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

        prompt = f"""
        你是一个专业的数据增强助手，为“指代表达分割”（Referring Image Segmentation）任务生成训练文本。
        目标：让{aug_num}条增强文本**尽可能不同，但指向同一对象**。

        ## 任务
        对以下 {len(texts)} 条指代表达，每条生成 **恰好 {aug_num} 条** 增强版本。必须：
        - 保持核心语义和所指对象**完全不变**
        - 不改变物体间空间关系（如左、右、上、中间等）
        - 可使用同义词替换，但不得损伤语义
        - 模仿 RefCOCO 数据集的自然、多样、口语化标注风格
        - 如果发现输入文本存在语法错误或者明显错误拼写，请根据修正后的文本进行增强,且不要解释任何错误以及留下修正的标记,如("(corrected)")

        ## 多样化要求（重点！）
        请尽可能从**不同角度、不同句式、不同表达习惯**生成两条差异明显的描述，例如：
        - 省略（如用“某个”、“..."）
        - 冗余（如加无害修饰：“正在...的”、“看起来像...”）
        - 使用不同主语视角（“那个...” vs “位于...的...”）
        - 调整语序（前置状语、后置定语）
        - 同义动词/形容词替换（“standing” → “located”、“near” → “close to”）
        - 句式变换：简单句 ↔ 复合句、主动 ↔ 被动（如适用）

        ## 语言要求
        - 永远使用英文进行增强

        ## 输出要求
        - 必须以 **简洁 JSON 格式** 输出
        - 使用字段 `"a"` 表示增强文本列表
        - 外层字段为 `"r"`（results）
        - 不要任何额外说明、markdown 或解释
        - 所有结果必须合并为一个 JSON 对象：{{"r": [...]}}，不要多个代码块
        - 严格生成 {len(texts)} 条和原文对应的增强文本json字符串，不可多也不可少

        ## 输入
        {input_section}

        ## 输出（必须是合法且最简的 JSON）：
        {{\"r\":[{{\"a\":[\"\",\"\"]}}]}}

        请发挥创造力，像人类标注员一样写出两条风格迥异但语义一致的描述！
        """

        for attempt in range(max_retries):
            try:
                response = await client.post(
                    BASE_URL.strip(),
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "top_p": TOP_P,
                        "frequency_penalty": 0.3,
                    },
                    timeout=TIMEOUT,
                )

                if response.status_code != 200:
                    print(f"❌ [Batch {batch_id}] HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue

                output = response.json()["choices"][0]["message"]["content"].strip()

                try:
                    data = repair_json(output, return_objects=True)
                    if not isinstance(data, dict):
                        raise ValueError("not dict")
                    results = data.get("r", [])
                    if len(results) != len(texts):
                        raise ValueError(f"数量不匹配: 期望 {len(texts)}, 得到 {len(results)}")
                    return results
                except Exception as e:
                    print(f"❌ [Batch {batch_id}] JSON 解析失败 (第 {attempt+1} 次): {e}")
                    print(f"📝 原始输出:\n{output[:300]}...\n")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            except Exception as e:
                print(f"❌ [Batch {batch_id}] 请求异常: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"💀 [Batch {batch_id}] 重试耗尽，放弃")

        return None  # 所有重试失败


# ======================
# 异步主函数：每批完成后立即保存 JSON（协程内保存版本）
# ======================
async def async_augment_pipeline(data_list: List[Dict[str, str]], output_dir: str):
    total = len(data_list)
    batches = [data_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"🚀 开始异步增强，共 {len(batches)} 批，每批 {BATCH_SIZE} 条...")
    print(f"⏱️  最大并发数: {MAX_CONCURRENT}")

    os.makedirs(output_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async with httpx.AsyncClient() as client:

        # 协程内部完成增强 + 保存
        async def process_batch_and_save(batch_data, batch_id):
            nonlocal success_count  # 引用外部计数器
            try:
                # 调用增强
                results = await async_batch_augment(
                    client=client,
                    items=batch_data,
                    batch_id=batch_id,
                    semaphore=semaphore,
                    max_retries=MAX_RETRIES,
                    aug_num=AUG_NUM
                )

                # 保存结果
                if results and len(results) == len(batch_data):
                    for item, aug_result in zip(batch_data, results):
                        try:
                            output_data = {
                                "original": item["text"],
                            }

                            for idx, aug_result in enumerate(aug_result["a"]):
                                output_data[f"augmented_{idx + 1}"] = aug_result

                            filename = f"{DATASET}_{SPLIT}_augtext_{item['idx']}.json"
                            filepath = os.path.join(output_dir, filename)
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                            success_count += 1
                        except Exception as e:
                            print(f"❌ 保存失败 {item['idx']}: {e}")
                else:
                    # 增强失败，仍保存空结构
                    for item in batch_data:
                        output_data = {
                            "original": item["text"],
                        }
                        for idx in range(AUG_NUM):
                            output_data[f"augmented_{idx + 1}"] = ""
                        filename = f"{DATASET}_{SPLIT}_augtext_{item['idx']}.json"
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                import traceback
                
                print(f"❌ 批次 {batch_id} 增强或保存失败: {traceback.format_exc()}")
                # 失败时仍为每条数据保存空文件
                for item in batch_data:
                    output_data = {
                        "original": item["text"],
                        "augmented_1": "",
                        "augmented_2": ""
                    }
                    filename = f"{DATASET}_{SPLIT}_augtext_{item['idx']}.json"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 创建任务
        tasks = [
            process_batch_and_save(batch, idx + 1)
            for idx, batch in enumerate(batches)
        ]

        # 使用 tqdm 异步等待所有任务完成
        success_count = 0
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="📦 增强 & 保存中"):
            await coro  # 等待每个协程完成（内部已保存）

    print(f"\n✅ 全部完成！共处理 {total} 条，成功保存 {success_count} 条")
    print(f"📁 已保存到目录: {output_dir}")

# ======================
# 主入口
# ======================
if __name__ == "__main__":
    data_root = f"{DATASET_ROOT}/{DATASET}/{SPLIT}_batch"

    # 1. 多线程加载前 500 条
    data_list = load_all_data(data_root, max_items=MAX_ITEMS)

    print(f"✅ 成功加载 {len(data_list)} 条数据")
    print("\n" + "=" * 60 + "\n")

    # 2. 异步增强（每个样本单独保存）
    asyncio.run(async_augment_pipeline(data_list, OUTPUT_DIR))