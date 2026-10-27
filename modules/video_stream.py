import cv2
import numpy as np
from typing import Generator, Any
from config import config

class VideoStream:
    """
    Класс для управления видеопотоком из файла.
    Использует контекстный менеджер для безопасного управления ресурсами OpenCV.
    """
    def __init__(self) -> None:
        self._source: str | int = config.video.source
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoStream":
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видеофайл: {self._source}")
        return self

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self._cap:
            while self._cap.isOpened():
                success, frame = self._cap.read()
                if not success:
                    break  # Видеофайл успешно закончился
                yield frame

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        if self._cap is not None:
            self._cap.release()
