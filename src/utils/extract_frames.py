"""
Extract frames from drone videos for dataset creation.

Usage:
    python extract_frames.py --video flight_01.mp4 --output frames/ --fps 1
"""
import cv2
import os
import argparse
from tqdm import tqdm


def extract_frames(video_path, output_dir, fps=1, resize=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / fps))
    print(f"[INFO] Video FPS     : {video_fps:.2f}")
    print(f"[INFO] Total frames  : {total}")
    print(f"[INFO] Extract every : {interval} frames")

    count, saved = 0, 0
    pbar = tqdm(total=total, desc="Extracting")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            name = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(name, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"[DONE] Saved {saved} frames to {output_dir}")
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True, help="Path to drone video")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--fps",    type=float, default=1,
                   help="Frames per second to extract")
    p.add_argument("--resize", type=int, nargs=2, default=None,
                   metavar=("W", "H"), help="Resize frames, e.g. 224 224")
    a = p.parse_args()
    extract_frames(a.video, a.output, a.fps,
                   tuple(a.resize) if a.resize else None)
