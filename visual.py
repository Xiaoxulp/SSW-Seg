import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from network.model12 import ConDSeg12

# ----------------------------
# 配置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = "/home/liuyameng/data/Glas/test/images/"  # 图片文件夹
save_dir = "/home/liuyameng/ConDSeg-main/results111"  # 保存 mask 图的文件夹
os.makedirs(save_dir, exist_ok=True)

# 模型
model = ConDSeg12()  # 你的模型实例
model.load_state_dict(torch.load("/home/liuyameng/ConDSeg-main/run_files/Glas/Glas_None_lr0.0001_20260102-191000/checkpoint.pth", map_location=device))
model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 根据你训练时大小调整
    transforms.ToTensor()
])


# ----------------------------
# 单张 mask 可视化函数
# ----------------------------
def save_mask(mask, orig_size, title, save_path, cmap):
    """
    mask: [C=1,H,W] Tensor
    orig_size: (H_orig, W_orig)
    """
    # 上采样到原图大小
    mask_resized = F.interpolate(mask.unsqueeze(0), size=orig_size, mode='bilinear', align_corners=True)
    mask_resized = mask_resized.squeeze().squeeze(0).cpu().numpy()  # ✅ 去掉 batch 和 channel 维 -> (H,W)

    # 阈值化
    mask_resized = (mask_resized > 0.5).astype(float)

    plt.figure(figsize=(4,4))
    plt.imshow(mask_resized, cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# ----------------------------
# 批量处理
# ----------------------------
for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 原图读取
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size[::-1]  # PIL (W,H) -> (H,W)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred, mask_fg, mask_bg, mask_uc = model(x)

    base_name = os.path.splitext(img_name)[0]

    # 保存三张 mask
    save_mask(mask_fg[0], orig_size, "Foreground", os.path.join(save_dir, f"{base_name}_fg.png"), cmap='Reds')
    save_mask(mask_bg[0], orig_size, "Background", os.path.join(save_dir, f"{base_name}_bg.png"), cmap='Blues')
    save_mask(mask_uc[0], orig_size, "Uncertainty", os.path.join(save_dir, f"{base_name}_uc.png"), cmap='Greens')

    print(f"Saved mask visualizations for {img_name}")
