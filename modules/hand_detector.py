from typing import Any, Dict, List
import numpy as np
import mediapipe as mp
from interfaces.detector import IDetector
import os
import urllib.request
from config import config

class HandDetector(IDetector):
    """
    Реализация детектора ключевых точек руки на базе современного Google MediaPipe 1.x (Vision Tasks API).
    Извлекает 21 нормализованную координату суставов кисти руки.
    """
    def __init__(self) -> None:
        self._model_path: str = config.mediapipe.model_path
        self._max_hands: int = config.mediapipe.max_num_hands
        self._min_detection_conf: float = config.mediapipe.min_detection_confidence
        self._min_tracking_conf: float = config.mediapipe.min_tracking_confidence
        self._detector: Any = None


    def load_model(self) -> None:
        """Инициализирует HandLandmarker, предварительно проверяя наличие файла весов."""
        import os
        import urllib.request

        # Автоматическое создание папок и скачивание модели, если её нет
        if not os.path.exists(self._model_path):
            print(f"[INFO] Файл модели не найден по пути: {self._model_path}")
            dir_name = os.path.dirname(self._model_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"[DOWNLOAD] Скачивание модели с Google Storage...")
            urllib.request.urlretrieve(url, self._model_path)
            print(f"[SUCCESS] Модель успешно скачана и сохранена в: {self._model_path}")

        # Стандартная инициализация MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._min_detection_conf,
            min_hand_presence_confidence=self._min_tracking_conf
        )
        
        self._detector = HandLandmarker.create_from_options(options)
        print("[INFO] Модель MediaPipe Hands 1.x (Tasks API) успешно инициализирована.")


    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает один кадр видеопотока.
        """
        if self._detector is None:
            raise RuntimeError("Детектор MediaPipe не загружен. Вызовите load_model().")

        # Переводим кадр из BGR (OpenCV) в RGB, как требует MediaPipe
        rgb_frame = frame[:, :, ::-1]
        
        # Конвертируем numpy-массив в нативный объект Image из MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Для режима RunningMode.VIDEO требуется монотонно растущий таймстамп (в миллисекундах).
        # Используем фейковый шаг времени на основе кадра (например, 33 мс на кадр для 30 FPS).
        # В реальном времени здесь использовался бы time.time_ns() // 1_000_000
        static_timestamp_ms = int(getattr(self, "_frame_counter", 0) * 33)
        setattr(self, "_frame_counter", getattr(self, "_frame_counter", 0) + 1)

        # Выполняем детекцию
        detection_result = self._detector.detect_for_video(mp_image, static_timestamp_ms)
        
        hand_landmarks_list: List[List[float]] = []

        # Извлекаем координаты, если найдена хотя бы одна рука
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            # Берем первую найденную руку
            first_hand_landmarks = detection_result.hand_landmarks[0]
            for lm in first_hand_landmarks:
                hand_landmarks_list.append([lm.x, lm.y, lm.z])

        return {"landmarks": hand_landmarks_list}
