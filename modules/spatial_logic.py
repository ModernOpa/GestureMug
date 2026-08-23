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
    Модуль бизнес-логики (Advanced Stage 5.2). 
    Определяет жест 'Шака' и использует Raycasting + фильтрацию по площади (Z-глубине)
    для однозначного определения целевого ID объекта в зашумленной сцене.
    """
    def __init__(self) -> None:
        self._hold_threshold: int = config.logic.trigger_hold_frames
        self._spatial_threshold: int = config.logic.spatial_threshold_px
        self._frame_counter: int = 0
        self._last_triggered_id: int = -1  # Храним ID кружки, над которой зафиксирован жест

    def _is_shaka_gesture(self, landmarks: List[List[float]]) -> bool:
        """Проверяет геометрию кисти на соответствие жесту 'Шака'."""
        if not landmarks or len(landmarks) < 21:
            return False

        lms = np.array(landmarks)
        wrist = lms[0]

        # Кончики и суставы
        thumb_tip = lms[4]
        thumb_ip = lms[3]  # Сустав большого пальца для вектора направления
        
        index_tip, index_pip = lms[8], lms[6]
        middle_tip, middle_pip = lms[12], lms[10]
        ring_tip, ring_pip = lms[16], lms[14]
        pinky_tip = lms[20]

        # Указательный, средний, безымянный согнуты к ладони
        index_folded = np.linalg.norm(index_tip - wrist) < np.linalg.norm(index_pip - wrist)
        middle_folded = np.linalg.norm(middle_tip - wrist) < np.linalg.norm(middle_pip - wrist)
        ring_folded = np.linalg.norm(ring_tip - wrist) < np.linalg.norm(ring_pip - wrist)

        # Большой и мизинец максимально разведены
        span_dist = np.linalg.norm(thumb_tip - pinky_tip)
        
        # Наклон большого пальца вниз (в OpenCV координата Y инвертирована: вниз - это плюс)
        thumb_pointing_down = thumb_tip[1] > thumb_ip[1]

        return bool(index_folded and middle_folded and ring_folded and span_dist > 0.15 and thumb_pointing_down)

    def check_trigger(self, yolo_results: Dict[str, Any], hand_results: Dict[str, Any]) -> bool:
        """
        Проверяет пространственно-временной триггер с использованием Raycasting и Z-глубины.
        Обновляет self._last_triggered_id при успешном матчинге.
        """
        boxes: List[List[Any]] = yolo_results.get("boxes", [])
        landmarks: List[List[float]] = hand_results.get("landmarks", [])

        if not landmarks or not self._is_shaka_gesture(landmarks):
            self._frame_counter = max(0, self._frame_counter - 1)
            self._last_triggered_id = -1
            return False

        lms = np.array(landmarks)
        thumb_ip_norm = lms[3]   # Сустав (начало луча)
        thumb_tip_norm = lms[4]  # Кончик (конец луча)

        # Переводим направляющие точки луча в пиксели экрана (1280x720)
        p1 = np.array([int(thumb_ip_norm[0] * 1280), int(thumb_ip_norm[1] * 720)])
        p2 = np.array([int(thumb_tip_norm[0] * 1280), int(thumb_tip_norm[1] * 720)])

        # Строим вектор луча направления большого пальца
        ray_vector = p2 - p1
        ray_length = np.linalg.norm(ray_vector)
        if ray_length == 0:
            return False
        ray_unit = ray_vector / ray_length

        candidate_cups: List[Dict[str, Any]] = []

        # Сканируем кружки (теперь извлекаем строго 6 параметров!)
        for box in boxes:
            x1, y1, x2, y2, conf, track_id = box
            
            # Считаем площадь Bounding Box (Эвристика Z-глубины: чем больше площадь, тем ближе объект)
            area = (x2 - x1) * (y2 - y1)

            # Проверяем Raycasting: протыкает ли луч прямоугольник кружки.
            # Шагаем по вектору луча вперед от кончика пальца на дистанцию до 300 пикселей
            intersect = False
            for step in range(0, 300, 10):
                check_point = p2 + ray_unit * step
                cx, cy = check_point[0], check_point[1]
                
                # Попадание в расширенную рамку кружки
                if (x1 - self._spatial_threshold <= cx <= x2 + self._spatial_threshold and 
                    y1 - self._spatial_threshold <= cy <= y2 + self._spatial_threshold):
                    intersect = True
                    break
            
            if intersect:
                candidate_cups.append({"id": track_id, "area": area})

        # Если луч пересек одну или несколько кружек
        if candidate_cups:
            # Сортируем кандидатов по площади (от большей к меньшей)
            # Самая большая кружка — на переднем плане. Берем её!
            candidate_cups.sort(key=lambda x: x["area"], reverse=True)
            best_target_id = candidate_cups[0]["id"]

            self._frame_counter += 1
            if self._frame_counter >= self._hold_threshold:
                self._frame_counter = 0  # Сброс триггера
                self._last_triggered_id = best_target_id  # Запоминаем целевой ID для БД
                return True
        else:
            self._frame_counter = max(0, self._frame_counter - 1)
            self._last_triggered_id = -1

        return False

    def get_last_triggered_id(self) -> int:
        """Возвращает ID кружки, которая вызвала последнее успешное срабатывание триггера."""
        return self._last_triggered_id
