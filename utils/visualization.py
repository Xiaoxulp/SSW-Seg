import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from network.model12 import ConDSeg12

# ----------------------------
# 配置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = "/home/liuyameng/data/Glas/test/labels/testA_9.png"  # 图片文件夹
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
def save_mask(mask, title, save_path, cmap):
    mask = mask.squeeze(0).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap=cmap)
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

    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred, mask_fg, mask_bg, mask_uc = model(x)

    # 保存三张 mask
    base_name = os.path.splitext(img_name)[0]
    save_mask(mask_fg[0], "Foreground", os.path.join(save_dir, f"{base_name}_fg.png"), cmap='Reds')
    save_mask(mask_bg[0], "Background", os.path.join(save_dir, f"{base_name}_bg.png"), cmap='Blues')
    save_mask(mask_uc[0], "Uncertainty", os.path.join(save_dir, f"{base_name}_uc.png"), cmap='Greens')

    print(f"Saved mask visualizations for {img_name}")
