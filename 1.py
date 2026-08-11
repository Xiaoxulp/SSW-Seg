import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from network.model import ConDSeg
from network.model12 import ConDSeg12

# ----------------------------
# 配置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = "/home/liuyameng/data/Glas/test/images/"
save_dir = "/home/liuyameng/ConDSeg-main/results111"
os.makedirs(save_dir, exist_ok=True)

# 模型
model = ConDSeg()
model.load_state_dict(
    torch.load("/home/liuyameng/ConDSeg-main/run_files/Glas/Glas_None_lr0.0001_20251224-002611/checkpoint.pth",
               map_location=device))
model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ----------------------------
# 改进的可视化函数
# ----------------------------
def save_mask_with_stats(mask, orig_size, title, save_path, cmap='viridis', threshold=0.5):
    """
    mask: [C=1,H,W] Tensor
    orig_size: (H_orig, W_orig)
    """
    # 上采样到原图大小
    mask_resized = F.interpolate(mask.unsqueeze(0), size=orig_size, mode='bilinear', align_corners=True)
    mask_resized = mask_resized.squeeze().squeeze(0).cpu().numpy()  # (H,W)

    # 打印统计信息
    print(f"{title}: shape={mask_resized.shape}, min={mask_resized.min():.4f}, "
          f"max={mask_resized.max():.4f}, mean={mask_resized.mean():.4f}, "
          f"std={mask_resized.std():.4f}")

    # 检查值是否合理
    if mask_resized.max() - mask_resized.min() < 0.01:
        print(f"  Warning: {title} values have very small range!")

    # 阈值化
    mask_binary = (mask_resized > threshold).astype(float)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 左侧：原始概率图
    im1 = axes[0].imshow(mask_resized, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(f"{title} (Probability)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 右侧：二值化结果
    im2 = axes[1].imshow(mask_binary, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"{title} (Binary, thr={threshold})")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 保存原始数据供进一步分析
    np.save(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(save_path))[0]}_raw.npy"), mask_resized)

    return mask_resized, mask_binary


# ----------------------------
# 批量处理（先测试一张）
# ----------------------------
img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 先测试第一张图片
if img_list:
    img_name = img_list[0]
    img_path = os.path.join(img_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size[::-1]  # PIL (W,H) -> (H,W)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask, mask_fg, mask_bg, mask_uc = model(x)

    base_name = os.path.splitext(img_name)[0]

    print("\n" + "=" * 50)
    print(f"Processing: {img_name}")
    print("=" * 50)

    # 保存并分析每个mask
    fg_prob, fg_bin = save_mask_with_stats(
        mask_fg[0], orig_size, "Foreground",
        os.path.join(save_dir, f"{base_name}_fg_detailed.png"),
        cmap='Reds'
    )

    bg_prob, bg_bin = save_mask_with_stats(
        mask_bg[0], orig_size, "Background",
        os.path.join(save_dir, f"{base_name}_bg_detailed.png"),
        cmap='Blues'
    )

    uc_prob, uc_bin = save_mask_with_stats(
        mask_uc[0], orig_size, "Uncertainty",
        os.path.join(save_dir, f"{base_name}_uc_detailed.png"),
        cmap='Greens'
    )

    main_prob, main_bin = save_mask_with_stats(
        mask[0], orig_size, "Main Prediction",
        os.path.join(save_dir, f"{base_name}_main_detailed.png"),
        cmap='viridis'
    )

    # 检查前景和背景的关系
    print("\n" + "=" * 50)
    print("Relationship Analysis:")
    print("=" * 50)

    # 计算前景+背景
    fg_plus_bg = fg_prob + bg_prob
    print(f"FG + BG - min: {fg_plus_bg.min():.4f}, max: {fg_plus_bg.max():.4f}, mean: {fg_plus_bg.mean():.4f}")

    # 差异分析
    diff_fg_bg = np.abs(fg_prob - (1 - bg_prob))
    print(f"|FG - (1-BG)| - max: {diff_fg_bg.max():.4f}, mean: {diff_fg_bg.mean():.4f}")

    # 创建关系图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # FG vs BG 散点图
    axes[0, 0].scatter(fg_prob.flatten(), bg_prob.flatten(), alpha=0.1, s=1)
    axes[0, 0].plot([0, 1], [1, 0], 'r--', alpha=0.5)  # 理想情况：FG + BG = 1
    axes[0, 0].set_xlabel('Foreground Probability')
    axes[0, 0].set_ylabel('Background Probability')
    axes[0, 0].set_title('FG vs BG Relationship')
    axes[0, 0].grid(True, alpha=0.3)

    # FG + BG 直方图
    axes[0, 1].hist(fg_plus_bg.flatten(), bins=50, alpha=0.7)
    axes[0, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('FG + BG')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sum of FG and BG')
    axes[0, 1].grid(True, alpha=0.3)

    # 不确定性分析
    axes[1, 0].hist(uc_prob.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Uncertainty Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 主预测直方图
    axes[1, 1].hist(main_prob.flatten(), bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Main Prediction')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Main Prediction Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_analysis.png"), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nDetailed analysis saved for {img_name}")