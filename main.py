import cv2
import os
from modules.video_stream import VideoStream
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.spatial_logic import SpatialLogicAnalyzer
from modules.mock_db import MockDatabaseClient

def run_stage4_self_test() -> None:
    """Система самопроверки (Smoke Test) для Этапа 4."""
    print("[TEST] Запуск самопроверки модуля пространственной логики...")
    analyzer = SpatialLogicAnalyzer()
    assert analyzer.check_trigger({"boxes": []}, {"landmarks": []}) is False
    mock_bad_hand = {"landmarks": [[0.0, 0.0, 0.0]] * 21}
    mock_cup = {"boxes": [[100, 100, 200, 200, 0.9]]}
    assert analyzer.check_trigger(mock_cup, mock_bad_hand) is False
    print("[TEST] Самопроверка Этапа 4 успешно пройдена!\n")


def main() -> None:
    print("=== ЗАПУСК СИСТЕМЫ КОМПЬЮТЕРНОГО ЗРЕНИЯ ===")
    
    try:
        run_stage4_self_test()
    except AssertionError as e:
        print(f"[CRITICAL] Тест логики провален: {e}")
        return

    # Инициализация всех модулей
    obj_detector = ObjectDetector()
    hand_detector = HandDetector()
    spatial_analyzer = SpatialLogicAnalyzer()
    db_client = MockDatabaseClient()  # Наш клиент базы данных
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
        db_client.connect()  # Подключаемся к БД
    except Exception as e:
        print(f"Ошибка при инициализации компонентов: {e}")
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
                
                # 1. Инференс моделей
                obj_results = obj_detector.process(frame)
                hand_results = hand_detector.process(frame)
                
                boxes = obj_results.get("boxes", [])
                landmarks = hand_results.get("landmarks", [])
                
                # 2. Отрисовка кружек (YOLO + ByteTrack)
                for box in boxes:
                    # Теперь у нас 6 параметров в коробке!
                    x1, y1, x2, y2, conf, track_id = box
                    
                    # Рисуем рамку вокруг кружки
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # HUD: Пишем конкретный ID: "Cup ID: 2 (0.92)"
                    label = f"Cup ID: {track_id} ({conf:.2f})"
                    cv2.putText(
                        frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # 3. Отрисовка суставов руки
                if landmarks:
                    for lm in landmarks:
                        pt_x = int(lm[0] * 1280)
                        pt_y = int(lm[1] * 720)
                        cv2.circle(frame, (pt_x, pt_y), 4, (0, 255, 0), -1)
                
                # 4. Анализ пространственного триггера (Raycasting + Z-глубина)
                is_triggered = spatial_analyzer.check_trigger(obj_results, hand_results)
                
                if is_triggered:
                    trigger_count += 1
                    
                    # Извлекаем точный ID кружки, выбранный нашей геометрической системой
                    target_cup_id = spatial_analyzer.get_last_triggered_id()
                    
                    print(f"[EVENT] Фиксация триггера над Cup ID: {target_cup_id} на кадре {frame_count}. Запись в БД...")
                    
                    # Отправляем событие в базу данных с точным ID объекта от ByteTrack
                    db_client.save_event(
                        event_type="Jumbo_Over_Cup",
                        metadata={
                            "frame_id": frame_count,
                            "target_object_id": target_cup_id,  # Уникальный ID от ByteTrack
                            "system_status": "SUCCESS"
                        }
                    )
                    
                    # HUD эффекты на видео кадра
                    cv2.putText(
                        frame, f"MATCH: JUMBO OVER CUP {target_cup_id}!", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
                    )
                    cv2.rectangle(frame, (0, 0), (1280, 720), (0, 255, 0), 5)
               
                video_writer.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Обработано кадров: {frame_count}")

            video_writer.release()
            print(f"\n[SUCCESS] Проект полностью реализован!")
            print(f"Всего событий записано в БД: {trigger_count}")
            print(f"Видеозапись сохранена в: {output_path}")
                    
    except RuntimeError as e:
        print(f"Критическая ошибка пайплайна: {e}")
    except Exception as e:
        print(f"Непредвиденное исключение: {e}")
    finally:
        # Гарантированно отключаемся от базы данных при любом исходе
        db_client.disconnect()

if __name__ == "__main__":
    main()
