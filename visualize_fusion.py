import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== 路径配置 ==========
BASE = r'D:\kitti_project\data\2011_09_26'
IMG_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'image_02', 'data')
VELO_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'velodyne_points', 'data')
CALIB_DIR = BASE

# ========== 读标定文件 ==========
def read_calib_cam_to_cam(filepath):
    """读取相机内参"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # P_rect_02: 左彩色相机的投影矩阵
    for line in lines:
        if line.startswith('P_rect_02:'):
            P = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            return P
    raise ValueError('P_rect_02 not found')

def read_calib_velo_to_cam(filepath):
    """读取激光雷达到相机的标定"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('R:'):
            R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        if line.startswith('T:'):
            T = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)
    return R, T

# ========== 读取数据 ==========
P2 = read_calib_cam_to_cam(os.path.join(CALIB_DIR, 'calib_cam_to_cam.txt'))
R, T = read_calib_velo_to_cam(os.path.join(CALIB_DIR, 'calib_velo_to_cam.txt'))

# 构建 4x4 变换: 激光雷达 → 相机 0
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :3] = R
Tr_velo_to_cam[:3, 3] = T.flatten()

# 读取第一帧
idx = 0
img = cv2.imread(os.path.join(IMG_DIR, f'{idx:010d}.png'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 读取点云 (.bin 格式: x, y, z, reflectance)
scan = np.fromfile(os.path.join(VELO_DIR, f'{idx:010d}.bin'), dtype=np.float32)
points = scan.reshape(-1, 4)

# 只保留相机前方的点 (z > 0)
points = points[points[:, 0] > 0]

# ========== 点云 → 图像投影 ==========
xyz = points[:, :3]               # (N, 3)
xyz_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])  # (N, 4)

# 变换: 激光雷达 → 相机坐标系
cam_xyz = (Tr_velo_to_cam @ xyz_homo.T).T[:, :3]  # (N, 3)

# 投影: 相机 → 像素
pixels = (P2 @ np.hstack([cam_xyz, np.ones((cam_xyz.shape[0], 1))]).T).T  # (N, 3)
pixels[:, 0] /= pixels[:, 2]   # u = x/z
pixels[:, 1] /= pixels[:, 2]   # v = y/z
depths = pixels[:, 2]          # 实际深度

u = pixels[:, 0].astype(int)
v = pixels[:, 1].astype(int)

# 过滤掉图像外的点
H, W = img.shape[:2]
mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, depths = u[mask], v[mask], depths[mask]

# ========== 可视化 ==========
# 归一化深度为颜色
depth_color = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)

# 生成彩色点
color_map = plt.cm.jet(depth_color)[:, :3] * 255
color_map = color_map.astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 左图：原始图像
axes[0].imshow(img)
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')

# 右图：点云投影
overlay = img.copy()
for i in range(len(u)):
    # 点越小越透明，越远越冷色
    alpha = max(0.2, 1.0 - depths[i] / 80)  # 距离越远越透明
    color = color_map[i].tolist()
    cv2.circle(overlay, (u[i], v[i]), 1, color, -1)

axes[1].imshow(overlay)
axes[1].set_title('LiDAR Points Projected', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE, '..', 'fusion_preview.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Done! 截图保存为 fusion_preview.png')