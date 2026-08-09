from typing import Any, Dict, List, cast
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from interfaces.detector import IDetector
from config import config

class ObjectDetector(IDetector):
    """
    Реализация детектора объектов на базе архитектуры YOLOv8/v11.
    Заточен под поиск и фильтрацию целевого класса (кружка) с использованием GPU.
    """
    def __init__(self) -> None:
        self._model_path: str = config.yolo.model_path
        self._conf_threshold: float = config.yolo.confidence_threshold
        self._device: str = config.yolo.device
        self._target_classes: tuple[int, ...] = config.yolo.target_classes
        self._model: YOLO | None = None

    def load_model(self) -> None:
        """Загружает веса YOLO и принудительно переносит модель на CUDA GPU."""
        import warnings
        # Подавляем предупреждение о совместимости архитектур CUDA (sm_120), так как инференс все равно работает корректно
        warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA capability.*")
        
        self._model = YOLO(self._model_path)
        self._model.to(self._device)
        print(f"[INFO] Модель YOLO успешно загружена на устройство: {self._device.upper()}")

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает кадр, находит объекты и фильтрует только класс 'cup'.
        """
        if self._model is None:
            raise RuntimeError("Модель YOLO не загружена. Вызовите load_model() перед обработкой.")

        raw_results = self._model(
            frame, 
            conf=self._conf_threshold, 
            device=self._device, 
            classes=list(self._target_classes),
            verbose=False
        )
        
        results_list = cast(List[Results], raw_results)
        detected_boxes: List[List[float]] = []
        
        if results_list and len(results_list) > 0:
            boxes_obj = cast(Boxes, results_list[0].boxes)
            
            # Senior-style извлечение: проверяем тип данных (Tensor или Array) перед конвертацией в numpy
            xyxy_raw = boxes_obj.xyxy
            conf_raw = boxes_obj.conf
            
            xyxy_np = xyxy_raw.cpu().numpy() if isinstance(xyxy_raw, torch.Tensor) else np.array(xyxy_raw)
            conf_np = conf_raw.cpu().numpy() if isinstance(conf_raw, torch.Tensor) else np.array(conf_raw)
            
            for i in range(len(xyxy_np)):
                coord = xyxy_np[i].tolist()  # [x1, y1, x2, y2]
                conf = float(conf_np[i])     # confidence score
                detected_boxes.append([*coord, conf])

        return {"boxes": detected_boxes}
