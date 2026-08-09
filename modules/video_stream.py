import cv2
import numpy as np
from typing import Generator
from config import config

class VideoStream:
    """
    Класс для управления видеопотоком (камера или файл).
    Использует протокол контекстного менеджера для безопасного управления ресурсами OpenCV.
    """
    def __init__(self) -> None:
        self._source: str | int = config.video.source
        self._width: int = config.video.width
        self._height: int = config.video.height
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoStream":
        """Открывает видеопоток и конфигурирует разрешение."""
        self._cap = cv2.VideoCapture(self._source)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видео: {self._source}")
        
        # Настройка разрешения (работает в основном для физических веб-камер)
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            
        return self

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Генератор, лениво возвращающий кадры из видеопотока.
        Предотвращает избыточное потребление оперативной памяти.
        """
        if self._cap is None:
            raise RuntimeError("Видеопоток не инициализирован. Используйте 'with VideoStream()'.")

        while self._cap.isOpened():
            success, frame = self._cap.read()
            if not success:
                # Если это файл — он закончился. Если камера — сбой кадра.
                break
            yield frame

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: float | None) -> None:
        """Гарантированно освобождает ресурсы камеры при выходе из контекста."""
        if self._cap is not None:
            self._cap.release()
        #cv2.destroyAllWindows()

