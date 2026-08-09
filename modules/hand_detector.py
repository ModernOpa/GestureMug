from typing import Any, Dict, List, cast
import cv2
import numpy as np
import mediapipe as mp
from interfaces.detector import IDetector
from config import config

class HandDetector(IDetector):
    """
    Реализация детектора ключевых точек руки на базе Google MediaPipe Hands.
    Извлекает 21 трехмерную координату суставов кисти.
    """
    def __init__(self) -> None:
        self._max_hands: int = config.mediapipe.max_num_hands
        self._min_detection_conf: float = config.mediapipe.min_detection_confidence
        self._min_tracking_conf: float = config.mediapipe.min_tracking_confidence
        
        # Инициализация API MediaPipe
        self._mp_hands = mp.solutions.hands
        self._hands: Any = None

    def load_model(self) -> None:
        """Инициализирует контекст MediaPipe Hands."""
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,  # Оптимизировано для обработки видеопотока
            max_num_hands=self._max_hands,
            min_detection_confidence=self._min_detection_conf,
            min_tracking_confidence=self._min_tracking_conf
        )
        print("[INFO] Модель MediaPipe Hands успешно инициализирована.")

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает кадр и извлекает нормализованные координаты ключевых точек.
        
        :param frame: Исходный кадр в формате BGR (OpenCV)
        :return: Словарь вида {'landmarks': [[x, y, z], ...]} для первой найденной руки
        """
        if self._hands is None:
            raise RuntimeError("Модель MediaPipe Hands не загружена. Вызовите load_model().")

        # MediaPipe строго требует формат изображения RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        hand_landmarks_list: List[List[float]] = []

        # Если на кадре обнаружены руки
        if results.multi_hand_landmarks:
            # Берем только первую руку (согласно нашей конфигурации max_num_hands=1)
            first_hand = results.multi_hand_landmarks[0]
            
            for lm in first_hand.landmark:
                # lm.x и lm.y нормализованы от 0.0 до 1.0 относительно размеров кадра
                # lm.z представляет глубину (расстояние от камеры)
                hand_landmarks_list.append([lm.x, lm.y, lm.z])

        return {"landmarks": hand_landmarks_list}
