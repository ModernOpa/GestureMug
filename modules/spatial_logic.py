# Правило жеста (Геометрия кисти):
#В жесте «Шака» большой палец (точка №4) и мизинец (точка №20) должны быть максимально выпрямлены и разведены.
# Указательный, средний и безымянный пальцы (кончики: точки 8, 12, 16) должны быть согнуты к ладони. Как это проверить без тригонометрии?
# Очень просто: расстояние от кончиков этих пальцев до основания ладони (точка №0) должно быть меньше, чем расстояние от суставов этих же пальцев до основания.

# Правило направления (Большой палец вниз):
# Чтобы палец «смотрел» вниз в кружку, координата y кончика большого пальца (точка №4) должна быть больше (ниже на экране),
# чем координата y его первого сустава (точка №3 или №2).

# Правило пространства (Матчинг с кружкой):
# Координаты (x, y) кончика большого пальца в пикселях должны находиться внутри прямоугольника (Bounding Box) кружки, 
# полученного от YOLO (или с небольшим допустимым зазором, который мы указали в config.py).

from typing import Any, Dict, List
import numpy as np
from config import config

class SpatialLogicAnalyzer:
    """
    Модуль бизнес-логики. Анализирует взаимное расположение ключевых точек руки
    и bounding box'ов объектов для фиксации контекстных жестов.
    """
    def __init__(self) -> None:
        self._hold_threshold: int = config.logic.trigger_hold_frames
        self._spatial_threshold: int = config.logic.spatial_threshold_px
        self._frame_counter: int = 0  # Счетчик кадров удержания жеста

    def _is_shaka_gesture(self, landmarks: List[List[float]]) -> bool:
        """
        Проверяет, сложены ли точки руки в жест 'Шака' (Большой и мизинец выпрямлены, остальные согнуты).
        """
        if not landmarks or len(landmarks) < 21:
            return False

        # Конвертируем в numpy для быстрых векторных расчетов
        lms = np.array(landmarks)

        # Основание ладони (Wrist)
        wrist = lms[0]

        # Кончики пальцев (Tips)
        thumb_tip = lms[4]
        index_tip = lms[8]
        middle_tip = lms[12]
        ring_tip = lms[16]
        pinky_tip = lms[20]

        # Суставы пальцев (PIP - Proximal Interphalangeal)
        index_pip = lms[6]
        middle_pip = lms[10]
        ring_pip = lms[14]

        # 1. Проверяем, что указательный, средний и безымянный согнуты
        # Расстояние от кончика до запястья должно быть меньше, чем от сустава до запястья
        index_folded = np.linalg.norm(index_tip - wrist) < np.linalg.norm(index_pip - wrist)
        middle_folded = np.linalg.norm(middle_tip - wrist) < np.linalg.norm(middle_pip - wrist)
        ring_folded = np.linalg.norm(ring_tip - wrist) < np.linalg.norm(ring_pip - wrist)

        # 2. Проверяем, что большой палец и мизинец разведены (расстояние между ними значительное)
        span_dist = np.linalg.norm(thumb_tip - pinky_tip)
        
        # 3. Наклон большого пальца вниз: y-координата кончика (4) ниже, чем у его основания (2)
        # В OpenCV координата Y растет сверху вниз
        thumb_pointing_down = thumb_tip[1] > lms[2][1]

        if index_folded and middle_folded and ring_folded and (span_dist > 0.15) and thumb_pointing_down:
            return True
            
        return False

    def check_trigger(self, yolo_results: Dict[str, Any], hand_results: Dict[str, Any]) -> bool:
        """
        Основной метод проверки пространственно-временного триггера.
        
        :return: True, если целевой жест удерживается над кружкой заданное количество кадров.
        """
        boxes: List[List[float]] = yolo_results.get("boxes", [])
        landmarks: List[List[float]] = hand_results.get("landmarks", [])

        # Если руки нет в кадре или жест не "Шака" — сбрасываем счетчик удержания
        if not landmarks or not self._is_shaka_gesture(landmarks):
            self._frame_counter = max(0, self._frame_counter - 1) # Плавное затухание вместо резкого сброса
            return False

        # Кончик большого пальца в нормализованных координатах
        thumb_tip_norm = landmarks[4]
        # Переводим в пиксельные координаты нашего стандарта (1280x720)
        thumb_x = int(thumb_tip_norm[0] * 1280)
        thumb_y = int(thumb_tip_norm[1] * 720)

        thumb_inside_cup = False

        # Проверяем пересечение с любой из найденных кружек
        for box in boxes:
            x1, y1, x2, y2, _ = box
            
            # Расширяем bounding box на допустимый пространственный порог (зазор)
            padded_x1 = x1 - self._spatial_threshold
            padded_y1 = y1 - self._spatial_threshold
            padded_x2 = x2 + self._spatial_threshold
            padded_y2 = y2 + self._spatial_threshold

            # Проверяем, попадает ли точка пальца внутрь рамки кружки
            if padded_x1 <= thumb_x <= padded_x2 and padded_y1 <= thumb_y <= padded_y2:
                thumb_inside_cup = True
                break

        if thumb_inside_cup:
            self._frame_counter += 1
            # Если жест стабильно удерживается нужное количество кадров
            if self._frame_counter >= self._hold_threshold:
                self._frame_counter = 0  # Сбрасываем счетчик, чтобы не спамить триггерами подряд
                return True
        else:
            self._frame_counter = max(0, self._frame_counter - 1)

        return False
