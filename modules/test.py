
"""test.py
Evaluation utilities: metrics and per-video aggregation.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_frame_level(model, X_test, y_test, batch_size=32):
    preds = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics, y_pred

def aggregate_frames_to_video(frame_preds, frames_per_video):
    """Simple aggregation: majority vote per video."""
    videos = []
    idx = 0
    for n in frames_per_video:
        vp = frame_preds[idx: idx+n]
        videos.append(int(np.round(np.mean(vp))))  # mean then round
        idx += n
    return videos
