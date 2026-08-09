import cv2
import os
from modules.video_stream import VideoStream
from modules.object_detector import ObjectDetector

def main() -> None:
    print("Инициализация компонентов системы...")
    
    # Инициализируем и загружаем YOLO
    detector = ObjectDetector()
    try:
        detector.load_model()
    except Exception as e:
        print(f"Ошибка загрузки модели YOLO: {e}")
        return

    try:
        with VideoStream() as stream:
            print("Пайплайн запущен. Обработка видеопотока...")
            
            # Проверяем, существует ли папка для сохранения
            output_dir = "data/video"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "output_processed.mp4")
            
            # Настройка VideoWriter для сохранения результата
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                30.0, 
                (1280, 720)
            )
            
            frame_count = 0
            for frame in stream.frames():
                frame_count += 1
                
                # Приводим к единому размеру
                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # Инференс детектора (YOLO на GPU)
                detection_results = detector.process(frame)
                boxes = detection_results.get("boxes", [])
                
                # Отрисовка рамок вокруг кружек
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"Cup: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # Записываем готовый кадр в файл
                video_writer.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Успешно обработано кадров: {frame_count}")
            
            # Закрываем файл
            video_writer.release()
            print(f"\n[SUCCESS] Этап 2 выполнен безупречно!")
            print(f"Результат сохранен в: {output_path}")
                    
    except RuntimeError as e:
        print(f"Критическая ошибка пайплайна: {e}")
    except Exception as e:
        print(f"Непредвиденное исключение: {e}")

if __name__ == "__main__":
    main()
