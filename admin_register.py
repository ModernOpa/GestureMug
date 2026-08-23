import os

import cv2

from modules.clip_encoder import CLIPFeatureExtractor
from modules.object_detector import ObjectDetector
from modules.vector_db import VectorRegistry
from modules.video_stream import VideoStream


def main() -> None:
    print("=== ПАНЕЛЬ АДМИНИСТРАТОРА: РЕГИСТРАЦИЯ ОБЪЕКТОВ ===")
    
    # Инициализируем компоненты
    yolo = ObjectDetector()
    clip = CLIPFeatureExtractor()
    v_db = VectorRegistry()
    
    try:
        yolo.load_model()
        clip.load_model()
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модели: {e}")
        return

    print("\n[INFO] Поиск объектов на видео для регистрации...")
    
    with VideoStream() as stream:
        for frame in stream.frames():
            # Шаг 1: Ищем кружки через YOLO
            yolo_results = yolo.process(frame)
            boxes = yolo_results.get("boxes", [])
            
            if not boxes:
                continue  # Ждем кадр, где кружка появится
                
            # Берем самую первую найденную кружку
            x1, y1, x2, y2, conf, track_id = boxes[0]
            
            # Шаг 2: Вырезаем (Crop) объект из кадра с помощью OpenCV
            # Добавляем небольшие проверки границ, чтобы не вылететь за пределы массива
            h, w, _ = frame.shape
            crop_y1, crop_y2 = max(0, int(y1)), min(h, int(y2))
            crop_x1, crop_x2 = max(0, int(x1)), min(w, int(x2))
            
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size == 0:
                continue
                
            # Показываем админу в консоли, что объект найден
            print(f"\n[FOUND] Обнаружен объект Cup (YOLO Conf: {conf:.2f})")
            print("Система готова извлечь цифровой паспорт (эмбеддинг) объекта.")
            
            # Запрашиваем имя у пользователяёё
            cup_name = input("Введите уникальное имя/владельца для этой кружки (например, 'Ivan_Mug'): ").strip()
            
            if not cup_name:
                print("[CANCEL] Регистрация отменена: пустое имя.")
                return
                
            # Шаг 3: Извлекаем 512-мерный вектор через OpenAI CLIP
            print("[COMPUTING] Извлечение признаков через CLIP на GPU...")
            embedding = clip.extract(crop)
            
            # Шаг 4: Записываем в векторную БД
            v_db.add_subscriber(cup_name, embedding)
            print(f"[SUCCESS] Объект '{cup_name}' успешно внесен в базу подписок!")
            break  # Выходим, так как объект успешно зарегистрирован

if __name__ == "__main__":
    main()
