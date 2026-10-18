import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class VideoConfig:
    source: str | int = "wsl_network"  # "data/video/video1.mp4"  # 0 для веб-камеры, или путь к 'video.mp4'
    width: int = 1280
    height: int = 720
    fps_target: int = 30
    output_dir: str = "data/video"
    output_filename: str = "output_processed.mp4"
    
@dataclass(frozen=True)
class NetworkConfig:
    host_ip: str = "192.168.182.20"
    cam_port: int = 9999
    ui_port: int = 9998
    display_env: str = ":0"

@dataclass(frozen=True)
class CLIPConfig:
    model_name: str = "clip-ViT-B-32"
    cache_dir: str = "data/neuro-models/clip-cache"

@dataclass(frozen=True)
class VectorDBConfig:
    db_path: str = "data/vector_db.json"
    reid_threshold: float = 0.77  # Порог сходства косинусного расстояния для подписки

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
class SoundConfig:
    sound_dir: str = "data/sounds"
    success_wav: str = "data/sounds/success.wav"
    denied_wav: str = "data/sounds/denied.wav"

@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    sounds: SoundConfig = field(default_factory=SoundConfig) 
    log_file_path: str = "data/events_log.txt"

# Единая точка доступа к конфигурации (Singleton)
config = AppConfig()
