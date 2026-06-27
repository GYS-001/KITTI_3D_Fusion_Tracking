import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========== 路径配置 ==========
BASE = r'D:\kitti_project\data\2011_09_26'
IMG_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'image_02', 'data')
VELO_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'velodyne_points', 'data')
CALIB_DIR = BASE
MODEL_PATH = r'D:\python_project\sam2-main\yolov8n.pt'

# ========== 读标定文件 ==========
def read_calib_cam_to_cam(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('P_rect_02:'):
            P = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            return P
    raise ValueError('P_rect_02 not found')

def read_calib_velo_to_cam(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('R:'):
            R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        if line.startswith('T:'):
            T = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)
    return R, T

# ========== 投影函数 ==========
def project_lidar_to_image(points_xyz, Tr_velo_to_cam, P2):
    """把激光雷达点云 (N,3) 投影到图像坐标"""
    xyz_homo = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
    cam_xyz = (Tr_velo_to_cam @ xyz_homo.T).T[:, :3]
    pixels = (P2 @ np.hstack([cam_xyz, np.ones((cam_xyz.shape[0], 1))]).T).T
    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]
    depths = pixels[:, 2]
    u = pixels[:, 0].astype(int)
    v = pixels[:, 1].astype(int)
    return u, v, depths

# ========== 载入数据 ==========
P2 = read_calib_cam_to_cam(os.path.join(CALIB_DIR, 'calib_cam_to_cam.txt'))
R, T = read_calib_velo_to_cam(os.path.join(CALIB_DIR, 'calib_velo_to_cam.txt'))
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :3] = R
Tr_velo_to_cam[:3, 3] = T.flatten()

# 选一帧（第 10 帧，场景里有几辆车）
idx = 10
img = cv2.imread(os.path.join(IMG_DIR, f'{idx:010d}.png'))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载点云
scan = np.fromfile(os.path.join(VELO_DIR, f'{idx:010d}.bin'), dtype=np.float32)
points = scan.reshape(-1, 4)
points = points[points[:, 0] > 0]          # 只保留相机前方
all_xyz = points[:, :3]

# 整个点云投影到图像
u_all, v_all, depths_all = project_lidar_to_image(all_xyz, Tr_velo_to_cam, P2)
H, W = img_rgb.shape[:2]
mask_all = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
u_all, v_all, depths_all, valid_xyz = u_all[mask_all], v_all[mask_all], depths_all[mask_all], all_xyz[mask_all]

# ========== YOLO 检测 ==========
model = YOLO(MODEL_PATH)
results = model(img_rgb, verbose=False)[0]

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
overlay = img_rgb.copy()

# 把有效点云画上去（半透明背景）
for i in range(0, len(u_all), 5):  # 采样画，不然太密
    depth_ratio = min(1.0, depths_all[i] / 60)
    r = int(255 * (1 - depth_ratio))
    b = int(255 * depth_ratio)
    cv2.circle(overlay, (u_all[i], v_all[i]), 1, (r, 0, b), -1)

# 画 2D 检测框
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f'{results.names[cls_id]} {conf:.2f}'
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

axes[0].imshow(img_rgb)
axes[0].set_title('Original + YOLO', fontsize=14)
axes[0].axis('off')

axes[1].imshow(overlay)
axes[1].set_title('LiDAR Projection + YOLO Boxes', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE, '..', 'fusion_detect.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Done! 截图保存为 fusion_detect.png')