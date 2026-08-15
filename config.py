import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(frozen=True)
class VideoConfig:
    source: str | int = "data/video/video1.mp4"  # 0 для веб-камеры, или путь к 'video.mp4'
    width: int = 1280
    height: int = 720
    fps_target: int = 30

@dataclass(frozen=True)
class YOLOConfig:
    model_path: str = "data/neuro-models/yolov8s.pt"  # Рекомендуется s (small) версия для баланса точности/скорости
    confidence_threshold: float = 0.5
    device: str = "cuda"  # Задействуем твою RTX 5060 Ti через WSL2
    target_classes: tuple[int, ...] = (41,)  # ID класса 'cup' в датасете COCO

@dataclass(frozen=True)
class MediaPipeConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    model_path: str = "data/hand-models/hand_landmarker.task"  # Путь к файлу модели

@dataclass(frozen=True)
class LogicConfig:
    trigger_hold_frames: int = 15  # Сколько кадров подряд нужно удерживать жест для триггера (0.5 сек при 30 FPS)
    spatial_threshold_px: int = 50  # Допустимый радиус/зазор между пальцем и bounding box кружки

@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    log_file_path: str = "data/events_log.txt"

# Единая точка доступа к конфигурации (Singleton)
config = AppConfig()
