import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer  # type: ignore

from config import config


class CLIPFeatureExtractor:
    """
    Экстрактор признаков на базе OpenAI CLIP (Metric Learning).
    Сжимает изображение объекта в уникальный числовой вектор (эмбеддинг) размерностью 512.
    """
    def __init__(self) -> None:
        self._device: str = config.yolo.device
        self._model_name: str = config.clip.model_name
        self._cache_dir: str = config.clip.cache_dir
        self._model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Загружает мультимодальную модель CLIP с сохранением в локальный кэш."""
        # cache_folder принудительно заставит скачать модель в data/neuro-models/clip-cache
        self._model = SentenceTransformer(
            self._model_name, 
            device=self._device, 
            cache_folder=self._cache_dir
        )
        print(f"[INFO] Модель OpenAI CLIP успешно загружена на устройство: {self._device.upper()}")


    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Принимает вырезанную область (OpenCV BGR), конвертирует в PIL Image 
        и генерирует нормализованный эмбеддинг.
        """
        if self._model is None:
            raise RuntimeError("Модель CLIP не инициализирована. Вызовите load_model().")

        if crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)

        # Конвертируем BGR (OpenCV) в RGB (стандарт PIL / CLIP)
        rgb_image = crop_bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb_image)

        # Извлекаем вектор признаков картинки
        embedding = self._model.encode(pil_img, convert_to_numpy=True, show_progress_bar=False)
        
        return embedding
