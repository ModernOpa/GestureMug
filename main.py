import cv2
import os
from modules.video_stream import VideoStream
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector

def main() -> None:
    print("Инициализация компонентов системы...")
    
    # Инициализация детектора объектов (YOLO)
    obj_detector = ObjectDetector()
    # Инициализация детектора рук (MediaPipe)
    hand_detector = HandDetector()
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
    except Exception as e:
        print(f"Ошибка при инициализации моделей: {e}")
        return

    try:
        with VideoStream() as stream:
            print("Пайплайн запущен. Обработка мультимодального видеопотока...")
            
            output_dir = "data/video"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output_processed.mp4")
            
            video_writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
                30.0, 
                (1280, 720)
            )
            
            frame_count = 0
            for frame in stream.frames():
                frame_count += 1
                
                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # --- ПОТОК А: Детекция кружек (YOLO на GPU) ---
                obj_results = obj_detector.process(frame)
                boxes = obj_results.get("boxes", [])
                
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"Cup: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # --- ПОТОК Б: Детекция ключевых точек руки (MediaPipe) ---
                hand_results = hand_detector.process(frame)
                landmarks = hand_results.get("landmarks", [])
                
                # Запускаем цикл по точкам, только если рука обнаружена в кадре
                if landmarks:
                    for lm in landmarks:
                        # lm[0] - x, lm[1] - y (оба нормализованы от 0 до 1)
                        pt_x = int(lm[0] * 1280)
                        pt_y = int(lm[1] * 720)
                        
                        # Рисуем зеленую точку на каждом суставе
                        cv2.circle(frame, (pt_x, pt_y), 4, (0, 255, 0), -1)

                
                # Записываем комбинированный кадр
                video_writer.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Обработано кадров: {frame_count}")
            
            video_writer.release()
            print(f"\n[SUCCESS] Этап 3 выполнен! Модели работают синхронно.")
            print(f"Результат с детекцией кружек и рук сохранен в: {output_path}")
                    
    except RuntimeError as e:
        print(f"Критическая ошибка пайплайна: {e}")
    except Exception as e:
        print(f"Непредвиденное исключение: {e}")

if __name__ == "__main__":
    main()
