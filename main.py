import os

import cv2

from modules.clip_encoder import CLIPFeatureExtractor
from modules.hand_detector import HandDetector
from modules.mock_db import MockDatabaseClient
from modules.object_detector import ObjectDetector
from modules.spatial_logic import SpatialLogicAnalyzer
from modules.vector_db import VectorRegistry
from modules.video_stream import VideoStream


def run_stage4_self_test() -> None:
    """Система smoke-тестирования геометрии."""
    print("[TEST] Запуск самопроверки модуля пространственной логики...")
    analyzer = SpatialLogicAnalyzer()
    assert analyzer.check_trigger({"boxes": []}, {"landmarks": []}) is False
    mock_bad_hand = {"landmarks": [[0.0, 0.0, 0.0]] * 21}
    mock_cup = {"boxes": [[100, 100, 200, 200, 0.9, 1]]} # Обновлено до 6 параметров для теста!
    assert analyzer.check_trigger(mock_cup, mock_bad_hand) is False
    print("[TEST] Самопроверка успешно пройдена!\n")


def main() -> None:
    print("=== ЗАПУСК МУЛЬТИМОДАЛЬНОЙ СИСТЕМЫ ВЕРИФИКАЦИИ ПОДПИСОК ===")
    
    try:
        run_stage4_self_test()
    except AssertionError as e:
        print(f"[CRITICAL] Тест логики провален: {e}")
        return

    # Инициализация абсолютно всех компонентов
    obj_detector = ObjectDetector()
    hand_detector = HandDetector()
    spatial_analyzer = SpatialLogicAnalyzer()
    clip_encoder = CLIPFeatureExtractor()
    vector_db = VectorRegistry()
    event_db = MockDatabaseClient()
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
        clip_encoder.load_model()
        event_db.connect()
    except Exception as e:
        print(f"Ошибка при инициализации компонентов: {e}")
        return

    try:
        with VideoStream() as stream:
            print("Пайплайн запущен. Ведется потоковый конвейерный анализ...")
            
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
                
                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                # 1. Сквозной инференс моделей базового уровня
                obj_results = obj_detector.process(frame)
                hand_results = hand_detector.process(frame)
                
                boxes = obj_results.get("boxes", [])
                landmarks = hand_results.get("landmarks", [])
                
                # 2. Визуализация HUD для кружек
                for box in boxes:
                    x1, y1, x2, y2, conf, track_id = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"Cup ID: {track_id}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                    )
                
                # 3. Визуализация HUD для рук
                if landmarks:
                    for lm in landmarks:
                        pt_x = int(lm[0] * 1280)
                        pt_y = int(lm[1] * 720)
                        cv2.circle(frame, (pt_x, pt_y), 4, (0, 255, 0), -1)
                
                # 4. Пространственно-временной анализ (Raycasting + Z-глубина)
                is_triggered = spatial_analyzer.check_trigger(obj_results, hand_results)
                
                if is_triggered:
                    trigger_count += 1
                    target_cup_id = spatial_analyzer.get_last_triggered_id()
                    
                    # Находим координаты целевой кружки по её ID для создания Crop'а
                    target_box = None
                    for box in boxes:
                        if box[5] == target_cup_id:
                            target_box = box
                            break
                    
                    owner_name = "Unknown (Unsubscribed)"
                    reid_score = 0.0
                    
                    # Если кружка найдена физически в кадре, запускаем ReID проверку подписки уровня Advanced
                    if target_box is not None:
                        x1, y1, x2, y2, _, _ = target_box
                        h, w, _ = frame.shape
                        crop_y1, crop_y2 = max(0, int(y1)), min(h, int(y2))
                        crop_x1, crop_x2 = max(0, int(x1)), min(w, int(x2))
                        
                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if crop.size > 0:
                            # Шаг А: Извлекаем эмбеддинг из вырезанного OpenCV куска через CLIP
                            current_embedding = clip_encoder.extract(crop)
                            # Шаг Б: Сравниваем с нашей векторной базой данных
                            reid_result = vector_db.verify_embedding(current_embedding)
                            
                            owner_name = reid_result["name"]
                            reid_score = reid_result["score"]
                    
                    print(f"[TRIGGER] Кадр {frame_count}: Жест над Cup ID {target_cup_id}. Владелец: {owner_name} (Score: {reid_score})")
                    
                    # Отправляем обогащенное событие в структурированную базу данных
                    event_db.save_event(
                        event_type="Contextual_Jumbo_Trigger",
                        metadata={
                            "frame_id": frame_count,
                            "track_id": target_cup_id,
                            "resolved_owner": owner_name,
                            "matching_confidence": reid_score,
                            "subscription_valid": owner_name != "Unknown (Unsubscribed)"
                        }
                    )
                    
                    # Спецэффекты дополненной реальности (AR HUD) на видео
                    hud_color = (0, 255, 0) if owner_name != "Unknown (Unsubscribed)" else (0, 0, 255)
                    cv2.putText(
                        frame, f"ACCESS: {owner_name.upper()}!", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, hud_color, 3
                    )
                    cv2.rectangle(frame, (0, 0), (1280, 720), hud_color, 5)
                
                video_writer.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Успешно обработано кадров: {frame_count}")
            
            video_writer.release()
            print(f"\n[SUCCESS] Сверхсложный конвейер ИИ успешно завершил работу!")
            print(f"Всего событий верифицировано и внесено в журнал: {trigger_count}")
            print(f"Видеозапись с AR-HUD сохранена в: {output_path}")
                    
    except RuntimeError as e:
        print(f"Критическая ошибка пайплайна: {e}")
    except Exception as e:
        print(f"Непредвиденное исключение в главном цикле: {e}")
    finally:
        event_db.disconnect()

if __name__ == "__main__":
    main()
