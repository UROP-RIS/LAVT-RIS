import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from bert.tokenization_bert import BertTokenizer
from bert.modeling_bert import BertModel
from lib import segmentation
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# ================================
# 1. 配置参数
# ================================
image_path = 'demo/1.png'
sentence = 'black man in yellow shirt'
# weights = './checkpoints/refcoco_pseudo.pth'
weights = './checkpoints/model_best_refcoco.pth'
# weights = "./checkpoints/refcoco_4cards.pth"
device = 'cuda:0'
output_dir = './demo'

# 输出路径
os.makedirs(output_dir, exist_ok=True)
weight_name = weights.split("/")[-1].split(".")[0]
output_path = os.path.join(output_dir, f'demo_result_{weight_name}_{sentence.replace(" ", "_")}.jpg')


# ================================
# 2. 图像预处理
# ================================
def load_and_transform_image(image_path, target_size=480):
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size  # (w, h)
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor, np.array(img_pil), original_size


img_tensor, img_np, (orig_w, orig_h) = load_and_transform_image(image_path)
img_tensor = img_tensor.to(device)


# ================================
# 3. 文本预处理（BERT Tokenization）
# ================================
def tokenize_sentence(sentence, max_length=20):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded = tokenizer.encode(sentence, add_special_tokens=True)[:max_length]
    padding_length = max_length - len(encoded)
    
    padded_ids = encoded + [0] * padding_length
    attention_mask = [1] * len(encoded) + [0] * padding_length
    
    return torch.tensor(padded_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)


sent_ids, attention_mask = tokenize_sentence(sentence)
sent_ids = sent_ids.to(device)
attention_mask = attention_mask.to(device)


# ================================
# 4. 模型定义与权重加载
# ================================
class Args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0


# 初始化模型
args = Args()
model = segmentation.__dict__['lavt'](pretrained='', args=args).to(device)
bert_model = BertModel.from_pretrained('bert/models').to(device)
bert_model.pooler = None  # 移除 pooler 层

# 加载检查点
checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'])
bert_model.load_state_dict(checkpoint['bert_model'])

model = model.to(device)
bert_model = bert_model.to(device)


# ================================
# 5. 推理
# ================================
with torch.no_grad():
    # BERT 编码文本
    last_hidden_states = bert_model(sent_ids, attention_mask=attention_mask)[0]  # (1, L, D)
    text_embedding = last_hidden_states.permute(0, 2, 1)  # (1, D, L)

    # 模型前向传播（输出为 logits）
    logits = model(img_tensor, text_embedding, l_mask=attention_mask.unsqueeze(-1))  # (1, 2, H, W)

    # 获取分割结果（argmax）
    pred_mask = logits.argmax(dim=1, keepdim=True)  # (1, 1, H, W)
    pred_mask = F.interpolate(pred_mask.float(), (orig_h, orig_w), mode='nearest')
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # (H, W)


# ================================
# 6. 可视化与保存
# ================================
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], alpha=0.4):
    """ 将分割 mask 叠加到原图上 """
    from scipy.ndimage import binary_dilation
    colors = np.array(colors) * 1.0
    overlay = image.copy()
    contours = binary_dilation(mask == 1) ^ (mask == 1)

    # 前景叠加
    overlay[mask == 1] = alpha * colors[1] + (1 - alpha) * overlay[mask == 1]
    # 轮廓加黑边
    overlay[contours] = 0
    return overlay.astype(np.uint8)


# 叠加并保存
visualization = overlay_davis(img_np, pred_mask)
result_image = Image.fromarray(visualization)
result_image.save(output_path)
print(f"结果已保存至: {output_path}")
# result_image.show()  # 可选：显示图像


# -------------------------------
# 7. 绘制带最大值标记的置信度热力图
# -------------------------------

# 1. 将 logits 转为概率（softmax）
probs = F.softmax(logits, dim=1)  # (1, 2, H, W)
foreground_prob = probs[0, 1, :, :]  # 取前景类概率 (H, W)
prob_np = foreground_prob.cpu().numpy()

# 2. 插值到原图大小
from torch.nn import functional as F_torch
prob_4d = foreground_prob.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
prob_resized = F_torch.interpolate(prob_4d, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
prob_resized = prob_resized.squeeze().cpu().numpy()  # (orig_h, orig_w)

# 3. 找到最大值的位置 (y, x)
max_confidence = prob_resized.max()
max_idx = np.unravel_index(prob_resized.argmax(), prob_resized.shape)
y_max, x_max = max_idx  # 注意：numpy 是 (行, 列) → (y, x)

print(f"最高置信度: {max_confidence:.3f}, 位置: (x={x_max}, y={y_max})")

# 4. 绘制热力图 + 标记最大值点
plt.figure(figsize=(orig_w / 100, orig_h / 100), dpi=100)
plt.imshow(prob_resized, cmap='jet')

# 在最大值处画一个红圈 + 十字
plt.plot(x_max, y_max, marker='o', color='white', markersize=6, markeredgewidth=2, markerfacecolor='none', markeredgecolor='red')
plt.plot(x_max, y_max, marker='+', color='red', markersize=10, markeredgewidth=2)

# 去掉坐标轴，保持紧凑
plt.axis('off')
plt.tight_layout(pad=0)

# 保存路径
heatmap_path = output_path.replace('.jpg', '_confidence_max_marked.png')
plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"带最大值标记的热力图已保存至: {heatmap_path}")