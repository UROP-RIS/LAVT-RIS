from transformers import pipeline
import torch

class BackTranslationAugmenter:
    """
    一个使用 Hugging Face Transformers 实现的回译数据增强器。
    支持批量处理文本。
    """
    def __init__(self, 
                 source_lang="en", 
                 intermediate_lang="fr", 
                 device=None):
        """
        初始化回译器。
        
        Args:
            source_lang (str): 源语言代码，例如 "en" (英语)
            intermediate_lang (str): 中间语言代码，例如 "fr" (法语), "de" (德语)
            device (str or torch.device): 指定运行设备。None 表示自动选择 (GPU if available)
        """
        self.source_lang = source_lang
        self.intermediate_lang = intermediate_lang
        
        # 自动选择设备
        if device is None:
            device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        
        # 构建模型名称
        model_name_fr2en = f"Helsinki-NLP/opus-mt-{intermediate_lang}-{source_lang}"
        model_name_en2fr = f"Helsinki-NLP/opus-mt-{source_lang}-{intermediate_lang}"
        
        print(f"正在加载回译模型 (从 {source_lang} -> {intermediate_lang} -> {source_lang})...")
        print(f"源到中间: {model_name_en2fr}")
        print(f"中间到源: {model_name_fr2en}")
        
        # 创建翻译 pipeline
        # 英语 -> 法语
        self.en_to_inter = pipeline(
            "translation", 
            model=model_name_en2fr, 
            device=device,
            # 可选参数，提高质量
            # max_length=512,  # 根据需要调整
            # num_beams=4,
            # early_stopping=True
        )
        
        # 法语 -> 英语
        self.inter_to_en = pipeline(
            "translation", 
            model=model_name_fr2en, 
            device=device,
            # max_length=512,
            # num_beams=4,
            # early_stopping=True
        )
        
        print("回译器加载完成！")

    def augment(self, text):
        """
        对单个文本进行回译增强。
        
        Args:
            text (str): 输入的英文文本。
            
        Returns:
            str: 回译后的英文文本，语义保持不变。
        """
        try:
            # Step 1: 英语 -> 中间语言 (e.g., 法语)
            intermediate_result = self.en_to_inter(text)
            intermediate_text = intermediate_result[0]['translation_text']
            
            # Step 2: 中间语言 -> 英语
            final_result = self.inter_to_en(intermediate_text)
            back_translated_text = final_result[0]['translation_text']
            
            return back_translated_text
            
        except Exception as e:
            print(f"回译失败，原文本: '{text}'。错误: {e}")
            # 如果失败，返回原文本作为兜底
            return text

    def augment_batch(self, texts):
        """
        对文本列表进行批量回译增强。
        
        Args:
            texts (list of str): 输入的英文文本列表。
            
        Returns:
            list of str: 回译后的英文文本列表。
        """
        augmented_texts = []
        for text in texts:
            augmented_text = self.augment(text)
            augmented_texts.append(augmented_text)
        return augmented_texts
    

if __name__ == "__main__":
    # 1. 初始化回译器
    # 你可以尝试不同的中间语言，如 'de' (德语), 'es' (西班牙语)
    bt_augmenter = BackTranslationAugmenter(source_lang="en", intermediate_lang="ru")
    
    # 2. 准备你的文本数据 (例如，RIS 任务中的 referring expressions)
    original_texts = [
        "a woman in a red dress standing on the left",
        "the dog sitting under the tree",
        "a car parked in front of the house",
        "the man with a blue hat near the window",
        "a group of people walking on the sidewalk"
    ]
    
    print("\n" + "="*60)
    print("回译数据增强结果:")
    print("="*60)
    
    # 3. 进行批量增强
    enhanced_texts = bt_augmenter.augment_batch(original_texts)
    
    # 4. 打印对比结果
    for orig, enh in zip(original_texts, enhanced_texts):
        print(f"原文本:  {orig}")
        print(f"回译后: {enh}")
        print("-" * 60)