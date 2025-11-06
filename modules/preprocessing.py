
"""preprocessing.py
Utilities for frame extraction, preprocessing, dataset creation.
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def extract_frames_from_video(video_path, frame_rate=1, max_frames=None):
    """Extract frames from a video at `frame_rate` frames per second (approx).
    Returns list of frames (as numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps / frame_rate), 1)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames

def preprocess_frame(frame, target_size=(224,224)):
    """Resize and normalize a single frame."""
    frame_resized = cv2.resize(frame, target_size)
    frame_norm = frame_resized.astype('float32') / 255.0
    return frame_norm

def create_dataset_from_videos(video_paths, labels, frame_rate=1, target_size=(224,224), max_frames_per_video=30):
    """Given lists of video file paths and labels, return X (numpy) and y (numpy).
    X shape: (num_samples, H, W, C) -- flattened frames across videos.
    y shape: (num_samples,)
    Note: This loads all frames into memory; for large datasets use generators.
    """
    X, y = [], []
    for vp, lb in zip(video_paths, labels):
        frames = extract_frames_from_video(vp, frame_rate=frame_rate, max_frames=max_frames_per_video)
        for f in frames:
            X.append(preprocess_frame(f, target_size=target_size))
            y.append(lb)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
