import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========== 路径配置 ==========
BASE = r'D:\kitti_project\data\2011_09_26'
IMG_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'image_02', 'data')
VELO_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'velodyne_points', 'data')
CALIB_DIR = BASE
MODEL_PATH = r'D:\python_project\sam2-main\yolov8n.pt'

# ========== 读标定 ==========
def read_calib_cam_to_cam(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                return np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

def read_calib_velo_to_cam(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('R:'): R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
        if line.startswith('T:'): T = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)
    return R, T

def project_lidar_to_image(points_xyz, Tr_velo_to_cam, P2):
    xyz_homo = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
    cam_xyz = (Tr_velo_to_cam @ xyz_homo.T).T[:, :3]
    pixels = (P2 @ np.hstack([cam_xyz, np.ones((cam_xyz.shape[0], 1))]).T).T
    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]
    return pixels[:, 0].astype(int), pixels[:, 1].astype(int), pixels[:, 2]

# ========== 载入 ==========
P2 = read_calib_cam_to_cam(os.path.join(CALIB_DIR, 'calib_cam_to_cam.txt'))
R, T = read_calib_velo_to_cam(os.path.join(CALIB_DIR, 'calib_velo_to_cam.txt'))
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :3] = R
Tr_velo_to_cam[:3, 3] = T.flatten()

idx = 10
img = cv2.imread(os.path.join(IMG_DIR, f'{idx:010d}.png'))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]

scan = np.fromfile(os.path.join(VELO_DIR, f'{idx:010d}.bin'), dtype=np.float32)
points = scan.reshape(-1, 4)
points = points[points[:, 0] > 0]
all_xyz = points[:, :3]

# ========== 全点云投影 ==========
u_all, v_all, d_all = project_lidar_to_image(all_xyz, Tr_velo_to_cam, P2)
mask = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
u_all, v_all, d_all, valid_xyz = u_all[mask], v_all[mask], d_all[mask], all_xyz[mask]

# ========== YOLO 检测 ==========
model = YOLO(MODEL_PATH)
results = model(img_rgb, verbose=False)[0]

# ========== 3D 框计算 ==========
vis = img_rgb.copy()
detections_3d = []

# 背景点云
for i in range(0, len(u_all), 8):
    ratio = min(1.0, d_all[i] / 60)
    cv2.circle(vis, (u_all[i], v_all[i]), 1,
               (int(255*(1-ratio)), 0, int(255*ratio)), -1)

for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = results.names[cls_id]

    if conf < 0.25:  # 过滤低置信度
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    # 稍微缩小检测框，去掉背景点
    margin = 0.15
    dx = (x2 - x1) * margin / 2
    dy = (y2 - y1) * margin / 2
    x1_i, y1_i = int(x1 + dx), int(y1 + dy)
    x2_i, y2_i = int(x2 - dx), int(y2 - dy)

    # 找到落在检测框内的点云
    in_box = (u_all >= x1_i) & (u_all < x2_i) & (v_all >= y1_i) & (v_all < y2_i)
    box_points = valid_xyz[in_box]

    if len(box_points) < 5:  # 点数太少，跳过
        continue

    # 统计 3D 信息
    center_3d = box_points.mean(axis=0)       # 物体中心
    size_3d = box_points.max(axis=0) - box_points.min(axis=0)  # 长宽高
    distance = np.linalg.norm(center_3d)       # 距离

    detections_3d.append({
        'label': label, 'conf': conf,
        'center': center_3d, 'size': size_3d,
        'distance': distance, 'num_points': len(box_points),
        'bbox_2d': (x1, y1, x2, y2)
    })

    # 画 2D 框 + 3D 信息
    color = (0, 255, 0) if cls_id == 2 else (255, 200, 0)  # 车绿色，其它黄色
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # 框内点云高亮
    for j in range(0, len(u_all), 1):
        if in_box[j]:
            cv2.circle(vis, (u_all[j], v_all[j]), 1, (0, 255, 255), -1)

    # 标注 3D 信息
    info_text = f'{label} {conf:.2f} | {distance:.1f}m | {int(size_3d[0]*100)}x{int(size_3d[1]*100)}x{int(size_3d[2]*100)}cm'
    cv2.putText(vis, info_text, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# ========== 终端打印 ==========
print(f'\n{"="*60}')
print(f'Frame {idx}: {len(detections_3d)} objects detected in 3D')
print(f'{"="*60}')
for d in detections_3d:
    c = d['center']
    s = d['size']
    print(f'  {d["label"]:8s} | conf={d["conf"]:.2f} | '
          f'dist={d["distance"]:.1f}m | '
          f'pos=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) | '
          f'size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})m | '
          f'pts={d["num_points"]}')

# ========== 显示 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.imshow(img_rgb)
ax1.set_title('Original + YOLO (2D)', fontsize=14)
ax1.axis('off')
for d in detections_3d:
    x1, y1, x2, y2 = d['bbox_2d']
    ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                 fill=False, edgecolor='lime', linewidth=2))

ax2.imshow(vis)
ax2.set_title('LiDAR Fusion + 3D Estimation', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE, '..', 'fusion_3dbox.png'), dpi=150, bbox_inches='tight')
plt.show()

# 3D 点云可视化
print('\nLaunching Open3D 3D viewer... (close window to continue)')
all_pcd = o3d.geometry.PointCloud()
all_pcd.points = o3d.utility.Vector3dVector(valid_xyz)
all_pcd.paint_uniform_color([0.6, 0.6, 0.6])

geometries = [all_pcd]
colors_3d = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
for i, d in enumerate(detections_3d):
    in_box = (u_all >= d['bbox_2d'][0]) & (u_all < d['bbox_2d'][2]) & \
             (v_all >= d['bbox_2d'][1]) & (v_all < d['bbox_2d'][3])
    obj_xyz = valid_xyz[in_box]
    if len(obj_xyz) < 5:
        continue
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_xyz)
    obj_pcd.paint_uniform_color(colors_3d[i % len(colors_3d)])

    bbox = obj_pcd.get_axis_aligned_bounding_box()
    bbox.color = colors_3d[i % len(colors_3d)]
    geometries.append(obj_pcd)
    geometries.append(bbox)

o3d.visualization.draw_geometries(geometries,
    window_name='3D Object Detection',
    width=1024, height=768,
    point_show_normal=False)