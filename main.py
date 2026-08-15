import cv2
import os
from modules.video_stream import VideoStream
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.spatial_logic import SpatialLogicAnalyzer

def run_stage4_self_test() -> None:
    """
    Система самопроверки (Smoke Test) для Этапа 4.
    Проверяет корректность работы геометрического триггера на синтетических данных.
    """
    print("[TEST] Запуск самопроверки модуля пространственной логики...")
    analyzer = SpatialLogicAnalyzer()
    
    # Сценарий 1: Пустые данные (не должно падать или выдавать True)
    assert analyzer.check_trigger({"boxes": []}, {"landmarks": []}) is False, "Ошибка: пустые данные вызвали триггер!"
    
    # Сценарий 2: Создаем синтетическую руку, которая НЕ является жестом 'Шака' (все точки в 0)
    mock_bad_hand = {"landmarks": [[0.0, 0.0, 0.0]] * 21}
    mock_cup = {"boxes": [[100, 100, 200, 200, 0.9]]}
    assert analyzer.check_trigger(mock_cup, mock_bad_hand) is False, "Ошибка: неверный жест вызвал триггер!"
    
    print("[TEST] Самопроверка Этапа 4 успешно пройдена! Математика стабильна.\n")


def main() -> None:
    print("Инициализация компонентов системы...")
    
    # Вызываем самопроверку перед запуском тяжелых моделей
    try:
        run_stage4_self_test()
    except AssertionError as e:
        print(f"[CRITICAL] Тест Этапа 4 провален: {e}")
        return

    # Инициализация модулей
    obj_detector = ObjectDetector()
    hand_detector = HandDetector()
    spatial_analyzer = SpatialLogicAnalyzer()
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
    except Exception as e:
        print(f"Ошибка при инициализации моделей: {e}")
        return

    try:
        with VideoStream() as stream:
            print("Пайплайн запущен. Анализ контекстных взаимодействий...")
            
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
            trigger_count = 0
            
            for frame in stream.frames():
                frame_count += 1
                
                if frame.shape != 1280 or frame.shape != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # 1. Инференс детекторов
                obj_results = obj_detector.process(frame)
                hand_results = hand_detector.process(frame)
                
                boxes = obj_results.get("boxes", [])
                landmarks = hand_results.get("landmarks", [])
                
                # 2. Отрисовка кружек (YOLO)
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"Cup: {conf:.2f}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # 3. Отрисовка суставов руки (MediaPipe)
                if landmarks:
                    for lm in landmarks:
                        # lm теперь гарантированно является списком из 3-х координат: [x, y, z]
                        pt_x = int(lm[0] * 1280)
                        pt_y = int(lm[1] * 720)
                        
                        # Рисуем красивую зеленую точку на каждом суставе
                        cv2.circle(frame, (pt_x, pt_y), 4, (0, 255, 0), -1)

                
                # 4. Проверка пространственно-временного триггера (Ядро проекта)
                is_triggered = spatial_analyzer.check_trigger(obj_results, hand_results)
                
                if is_triggered:
                    trigger_count += 1
                    print(f"[EVENT] Жест 'Шака' зафиксирован над кружкой на кадре {frame_count}! (Всего событий: {trigger_count})")
                    
                    # Визуальный HUD-эффект: рисуем жирный зеленый текст оповещения на видео кадра
                    cv2.putText(
                        frame, "MATCH: JUMBO OVER CUP!", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
                    )
                    # Подсвечиваем весь кадр зеленой рамкой
                    cv2.rectangle(frame, (0, 0), (1280, 720), (0, 255, 0), 5)
                
                # Записываем кадр в итоговый видеофайл
                video_writer.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Обработано кадров: {frame_count}")
            
            video_writer.release()
            print(f"\n[SUCCESS] Этап 4 завершен! Конвейер полностью собран.")
            print(f"Всего зафиксировано триггеров: {trigger_count}")
            print(f"Результат сохранен в: {output_path}")
                    
    except RuntimeError as e:
        print(f"Критическая ошибка пайплайна: {e}")
    except Exception as e:
        print(f"Непредвиденное исключение: {e}")

if __name__ == "__main__":
    main()
