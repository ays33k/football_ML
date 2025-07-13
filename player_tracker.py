# Player Tracking with Persistent IDs

import cv2, os
import numpy as np
from ultralytics import YOLO

# 2) Verify files
for f in ("best.pt","15sec_input_720p.mp4"):
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}")

# 3) Load model (classes: 0=ball,1=goalkeeper,2=player,3=referee)
model = YOLO("best.pt")

# 4) Video I/O
cap = cv2.VideoCapture("15sec_input_720p.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w, h))

# 5) Tracker state
tracks = []    # list of {'id':int, 'bbox':[x1,y1,x2,y2], 'missed':int}
next_id = 0
IOU_THRESHOLD = 0.3
MAX_MISSED = 5   # drop track after 5 missed frames

def compute_iou(a,b):
    xA = max(a[0],b[0]); yA = max(a[1],b[1])
    xB = min(a[2],b[2]); yB = min(a[3],b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

# 6) Process frames
while True:
    ret, frame = cap.read()
    if not ret: break

    # 6a) Detection
    res = model(frame)[0]
    dets = []
    for bx in res.boxes:
        if int(bx.cls[0])==2:  # player only
            dets.append(list(map(int, bx.xyxy[0])))

    # 6b) Build IoU matrix
    N, M = len(tracks), len(dets)
    iou_mat = np.zeros((N, M), dtype=np.float32)
    for i, tr in enumerate(tracks):
        for j, db in enumerate(dets):
            iou_mat[i,j] = compute_iou(tr['bbox'], db)

    # 6c) Greedy match
    matched_tracks = set()
    matched_dets   = set()
    while True:
        if iou_mat.size==0: break
        i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        if iou_mat[i,j] < IOU_THRESHOLD: break
        # assign
        tracks[i]['bbox'] = dets[j]
        tracks[i]['missed'] = 0
        matched_tracks.add(i)
        matched_dets.add(j)
        # zero out row&col
        iou_mat[i,:] = -1
        iou_mat[:,j] = -1

    # 6d) Create new tracks for unmatched detections
    for j, db in enumerate(dets):
        if j not in matched_dets:
            tracks.append({'id': next_id, 'bbox': db, 'missed': 0})
            next_id += 1

    # 6e) Increment missed count for unmatched tracks
    new_tracks = []
    for i, tr in enumerate(tracks):
        if i not in matched_tracks:
            tr['missed'] += 1
        if tr['missed'] <= MAX_MISSED:
            new_tracks.append(tr)
    tracks = new_tracks

    # 6f) Draw
    for tr in tracks:
        x1,y1,x2,y2 = tr['bbox']
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f"ID {tr['id']}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    out.write(frame)

# 7) Cleanup
cap.release()
out.release()
print("Done — output saved as 'output.mp4'")