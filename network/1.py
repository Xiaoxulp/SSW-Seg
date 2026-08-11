import albumentations as A
print(A.__version__)  # 确认 2.0.8

# 测试是否支持新参数
cd = A.CoarseDropout(
    p=0.3,
    num_holes_range=(1, 10),       # 替代 max_holes
    hole_height_range=(1, 32),     # 替代 max_height
    hole_width_range=(1, 32)       # 替代 max_width
    # 注意：此处移除了 fill_mode 和 fill_value 参数
)