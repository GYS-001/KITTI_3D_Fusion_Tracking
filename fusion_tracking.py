import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ========== Kalman 滤波器（恒定速度模型）==========
class KalmanTracker:
    def __init__(self, init_pos, track_id):
        self.track_id = track_id
        self.kf = cv2.KalmanFilter(6, 3)  # 状态6维，观测3维
        self.kf.transitionMatrix = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.array([*init_pos, 0,0,0], np.float32)
        self.history = [init_pos.copy()]
        self.age = 0
        self.hits = 1
        self.missed = 0

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        return pred[:3].flatten()

    def update(self, measurement):
        self.kf.correct(np.array(measurement, np.float32))
        pos = self.kf.statePost[:3].flatten()
        self.history.append(pos.copy())
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

# ========== 路径配置 ==========
BASE = r'D:\kitti_project\data\2011_09_26'
IMG_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'image_02', 'data')
VELO_DIR = os.path.join(BASE, '2011_09_26_drive_0001_sync', 'velodyne_points', 'data')
CALIB_DIR = BASE
MODEL_PATH = r'D:\python_project\sam2-main\yolov8n.pt'

# ========== 标定 ==========
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

def associate_detections_to_tracks(detections, trackers, iou_threshold=0.5):
    """用匈牙利算法做 3D 距离匹配"""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,))

    cost = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost[d, t] = np.linalg.norm(det['center'] - trk.kf.statePost[:3])

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = []
    unmatched_det = []
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched_det.append(d)

    unmatched_trk = []
    for t in range(len(trackers)):
        if t not in col_ind:
            unmatched_trk.append(t)

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > iou_threshold:  # 距离超过阈值不匹配
            unmatched_det.append(r)
            unmatched_trk.append(c)
        else:
            matched.append([r, c])

    return np.array(matched), np.array(unmatched_det), np.array(unmatched_trk)

# ========== 载入数据 ==========
P2 = read_calib_cam_to_cam(os.path.join(CALIB_DIR, 'calib_cam_to_cam.txt'))
R, T = read_calib_velo_to_cam(os.path.join(CALIB_DIR, 'calib_velo_to_cam.txt'))
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[:3, :3] = R
Tr_velo_to_cam[:3, 3] = T.flatten()

model = YOLO(MODEL_PATH)

# ========== 多帧跟踪 ==========
frame_indices = range(0, 30)  # 处理前30帧
trackers = []                 # 活跃跟踪器
next_id = 0
all_tracks_history = {}       # 记录所有轨迹（用于画图）

print(f'Processing {len(frame_indices)} frames...')

for fidx, frame_id in enumerate(frame_indices):
    img = cv2.imread(os.path.join(IMG_DIR, f'{frame_id:010d}.png'))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    scan = np.fromfile(os.path.join(VELO_DIR, f'{frame_id:010d}.bin'), dtype=np.float32)
    points = scan.reshape(-1, 4)
    points = points[points[:, 0] > 0]
    all_xyz = points[:, :3]

    u_all, v_all, d_all = project_lidar_to_image(all_xyz, Tr_velo_to_cam, P2)
    mask = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
    u_all, v_all, d_all, valid_xyz = u_all[mask], v_all[mask], d_all[mask], all_xyz[mask]

    # YOLO 检测
    results = model(img_rgb, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if float(box.conf[0]) < 0.25:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        margin = 0.15
        dx = (x2 - x1) * margin / 2
        dy = (y2 - y1) * margin / 2
        x1_i, y1_i = int(x1 + dx), int(y1 + dy)
        x2_i, y2_i = int(x2 - dx), int(y2 - dy)

        in_box = (u_all >= x1_i) & (u_all < x2_i) & (v_all >= y1_i) & (v_all < y2_i)
        box_points = valid_xyz[in_box]
        if len(box_points) < 5:
            continue

        detections.append({
            'center': box_points.mean(axis=0),
            'label': results.names[int(box.cls[0])],
            'conf': float(box.conf[0]),
            'bbox_2d': (x1, y1, x2, y2)
        })

    # ① 所有跟踪器预测
    for trk in trackers:
        trk.predict()

    # ② 匹配
    matched, unmatched_det, unmatched_trk = associate_detections_to_tracks(
        detections, trackers, iou_threshold=3.0)

    # ③ 更新匹配的
    for d_idx, t_idx in matched:
        trackers[t_idx].update(detections[d_idx]['center'])

    # ④ 未匹配的检测 → 新建跟踪器
    for d_idx in unmatched_det:
        trk = KalmanTracker(detections[d_idx]['center'], next_id)
        trackers.append(trk)
        all_tracks_history[next_id] = {'label': detections[d_idx]['label'],
                                        'history': trk.history.copy()}
        next_id += 1

    # ⑤ 未匹配的跟踪器标记丢失
    for t_idx in unmatched_trk:
        trackers[t_idx].mark_missed()

    # ⑥ 清理长期丢失的
    trackers = [trk for trk in trackers if trk.missed < 5]
    for trk in trackers:
        if trk.track_id in all_tracks_history:
            all_tracks_history[trk.track_id]['history'] = trk.history.copy()

    if fidx % 5 == 0:
        print(f'  Frame {frame_id}: {len(trackers)} active tracks')

# ========== 最终帧可视化 ==========
last_idx = frame_indices[-1]
img = cv2.imread(os.path.join(IMG_DIR, f'{last_idx:010d}.png'))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

vis = img_rgb.copy()
colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0),
          (255,0,255), (0,255,255), (255,128,0), (128,0,255)]

for tid, track_data in all_tracks_history.items():
    if len(track_data['history']) < 3:
        continue
    color = colors[tid % len(colors)]

    # 画当前估计位置
    cur_pos = track_data['history'][-1]
    u_c, v_c, _ = project_lidar_to_image(cur_pos.reshape(1,3), Tr_velo_to_cam, P2)
    if 0 <= u_c[0] < W and 0 <= v_c[0] < H:
        cv2.circle(vis, (u_c[0], v_c[0]), 8, color, -1)
        cv2.putText(vis, f'ID:{tid}', (u_c[0]+10, v_c[0]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 画历史轨迹
    pts = []
    for pos in track_data['history']:
        u, v, _ = project_lidar_to_image(pos.reshape(1,3), Tr_velo_to_cam, P2)
        if 0 <= u[0] < W and 0 <= v[0] < H:
            pts.append((u[0], v[0]))
    if len(pts) > 1:
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], color, 2)

# ========== 终端摘要 ==========
print(f'\n{"="*60}')
print(f'Tracking Complete: {len(all_tracks_history)} unique objects tracked')
print(f'{"="*60}')
for tid, data in all_tracks_history.items():
    if len(data['history']) >= 3:
        print(f'  ID:{tid}  {data["label"]:8s}  '
              f'tracked for {len(data["history"])} frames  '
              f'distance: {np.linalg.norm(data["history"][-1]):.1f}m')

plt.figure(figsize=(14, 6))
plt.imshow(vis)
plt.title(f'3D Multi-Object Tracking (Frame {frame_indices[0]}-{last_idx})', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(BASE, '..', 'fusion_tracking.png'), dpi=150, bbox_inches='tight')
plt.show()

print('\nDone! 截图保存为 fusion_tracking.png')