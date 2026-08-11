import torch
from torchvision import transforms
from PIL import Image
import os

from network.model12 import ConDSeg12
from utils.visualization import visualize_auxiliary_masks

# ------------------------
# 配置
img_path = "/home/liuyameng/data/Glas/test/images/testA_11.png"
weights_path = "/home/liuyameng/ConDSeg-main/run_files/Glas/Glas_None_lr0.0001_20260102-191000/checkpoint.pth"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# 读取图片
img_pil = Image.open(img_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
img = transform(img_pil).unsqueeze(0).to(device)

# ------------------------
# 加载模型
model = ConDSeg12().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# ------------------------
# 前向推理
with torch.no_grad():
    mask, mask_fg, mask_bg, mask_uc = model(img)

# ------------------------
# 可视化保存辅助 mask
visualize_auxiliary_masks(
    img.cpu(),
    mask_fg.cpu(),
    mask_bg.cpu(),
    mask_uc.cpu(),
    alpha=0.5,
    save_path=os.path.join(save_dir, "overlay_auxiliary.png"),
    batch_idx=0
)

# 可视化最终 mask
from utils.visualization import visualize_auxiliary_masks  # 如果有单 mask 可视化函数
visualize_auxiliary_masks(
    img.cpu(),
    mask_fg.cpu(),
    mask_bg.cpu(),
    mask_uc.cpu(),
    alpha=0.5,
    save_path=os.path.join(save_dir, "overlay_auxiliary.png"),
    batch_idx=0
)

print("✅ 可视化完成，结果保存在:", save_dir)
