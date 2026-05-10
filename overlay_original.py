import sys
import os
import argparse
import numpy as np
import cv2
import yaml

def show2Dpose_original(kps, img, offset_x, offset_y):
    # H36M format
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)   # Blue
    rcolor = (0, 0, 255)   # Red
    thickness = 3

    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        
        # Add offset
        sx, sy = int(start[0] + offset_x), int(start[1] + offset_y)
        ex, ey = int(end[0] + offset_x), int(end[1] + offset_y)
        
        cv2.line(img, (sx, sy), (ex, ey), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (sx, sy), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (ex, ey), thickness=-1, color=(0, 255, 0), radius=3)

    return img

def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    """在影像上畫虛線。"""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash_len if drawing else gap_len
        end_pos = min(pos + seg, length)
        if drawing:
            x1 = int(pt1[0] + ux * pos)
            y1 = int(pt1[1] + uy * pos)
            x2 = int(pt1[0] + ux * end_pos)
            y2 = int(pt1[1] + uy * end_pos)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        pos = end_pos
        drawing = not drawing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_video', required=True)
    parser.add_argument('--offsets_npz', required=True)
    parser.add_argument('--kps_npz', required=True)
    parser.add_argument('--config_yaml', default=None)
    parser.add_argument('--config_json', default=None)
    parser.add_argument('--output_video', required=True)
    args = parser.parse_args()

    if args.config_yaml:
        # Read config for start_line and end_line
        with open(args.config_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif args.config_json:
        import json
        config = json.loads(args.config_json)
    else:
        print("Error: Either --config_yaml or --config_json must be provided.")
        sys.exit(1)
    
    cameras = config.get('cameras', [])
    start_line = None
    end_line = None
    for cam in cameras:
        # Assuming the first valid camera configuration applies
        if cam.get('video_path') and os.path.abspath(cam['video_path']) == os.path.abspath(args.orig_video):
            start_line = cam.get('start_line')
            end_line = cam.get('end_line')
            break
            
    if start_line is None and end_line is None and len(cameras) > 0:
        start_line = cameras[0].get('start_line')
        end_line = cameras[0].get('end_line')

    # Read offsets
    offsets_data = np.load(args.offsets_npz)
    offsets = offsets_data['offsets']  # shape: (N, 2)
    orig_frames = offsets_data['orig_frames']  # shape: (N,)

    # Read keypoints
    kps_data = np.load(args.kps_npz, allow_pickle=True)
    keypoints = kps_data['reconstruction'][0]  # shape: (valid_frames, 17, 4)
    valid_frames = np.asarray(kps_data['valid_frames']).flatten().astype(int)

    # valid_frames represents the indices IN THE CROPPED VIDEO
    # orig_frames represents the mapping from CROPPED VIDEO index to ORIGINAL VIDEO index
    # We will output EXACTLY len(orig_frames) frames.
    
    # Create a mapping from v_idx (cropped frame index) to kps
    kps_map = {}
    for i, v_idx in enumerate(valid_frames):
        if v_idx < len(orig_frames):
            kps_map[v_idx] = keypoints[i]

    cap = cv2.VideoCapture(args.orig_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    print(f"Creating uncropped overlay video: {args.output_video} ({len(orig_frames)} frames)")
    from tqdm import tqdm
    
    with tqdm(total=len(orig_frames), desc="Overlaying") as pbar:
        for v_idx, orig_idx in enumerate(orig_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Draw start line and end line
            if start_line and end_line:
                p0 = (int(start_line[0][0]), int(start_line[0][1]))
                p3 = (int(start_line[1][0]), int(start_line[1][1]))
                p1 = (int(end_line[0][0]), int(end_line[0][1]))
                p2 = (int(end_line[1][0]), int(end_line[1][1]))
                
                # 四邊形虛線連接（上連上、下連下）
                for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
                    _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)
                
                # 實線標記起點與終點
                cv2.line(frame, p0, p3, (0, 0, 0), 5)
                cv2.line(frame, p0, p3, (180, 255, 255), 3) # yellow
                cv2.line(frame, p1, p2, (0, 0, 0), 5)
                cv2.line(frame, p1, p2, (255, 200, 100), 3) # light blue
            elif start_line:
                pt1 = (int(start_line[0][0]), int(start_line[0][1]))
                pt2 = (int(start_line[1][0]), int(start_line[1][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
                cv2.line(frame, pt1, pt2, (180, 255, 255), 3) # yellow
            elif end_line:
                pt1 = (int(end_line[0][0]), int(end_line[0][1]))
                pt2 = (int(end_line[1][0]), int(end_line[1][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
                cv2.line(frame, pt1, pt2, (255, 200, 100), 3) # light blue
                
            # Draw skeleton
            if v_idx in kps_map:
                kps = kps_map[v_idx]
                off_x, off_y = offsets[v_idx]
                frame = show2Dpose_original(kps, frame, off_x, off_y)
                
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Uncropped overlay video saved to {args.output_video}")

if __name__ == '__main__':
    main()
