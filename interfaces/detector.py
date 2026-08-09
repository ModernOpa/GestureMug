from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class IDetector(ABC):
    """
    Абстрактный интерфейс для всех модулей компьютерного зрения (YOLO, MediaPipe и т.д.).
    Гарантирует единообразную инициализацию и инференс моделей.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Инициализирует модель, выделяет память под веса и переносит вычисления на указанный девайс (CPU/GPU).
        """
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает один кадр изображения.
        
        :param frame: Исходный кадр в формате BGR (OpenCV)
        :return: Словарь со стандартизированными результатами детекции/трекинга
        """
        pass
