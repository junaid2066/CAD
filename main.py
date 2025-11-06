
"""main.py
Example entrypoint to preprocess, train, evaluate and plot results.
Adjust dataset paths and parameters as needed.
"""
import os
import numpy as np
from modules.preprocessing import create_dataset_from_videos, train_val_split
from modules.train import train_model
from modules.test import evaluate_frame_level
from modules.plot_results import plot_history, plot_confusion

def example_pipeline(real_video_paths, real_labels):
    # Create dataset (frames)
    X, y = create_dataset_from_videos(real_video_paths, real_labels, frame_rate=1, target_size=(224,224), max_frames_per_video=30)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)
    model, history = train_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=10)
    metrics, y_pred = evaluate_frame_level(model, X_val, y_val)
    print('Evaluation metrics:', metrics)
    plot_history(history)
    plot_confusion(metrics['confusion_matrix'])
    return model, history, metrics

if __name__ == '__main__':
    # Placeholder: user should populate real paths and labels
    real_video_paths = []
    real_labels = []
    print('Please edit main.py to point to your dataset paths and labels, then run the pipeline.')
