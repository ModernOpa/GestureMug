import cv2
import os
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, cast
import pygame  # type: ignore

from config import config
from modules.video_stream import VideoStream
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.spatial_logic import SpatialLogicAnalyzer
from modules.clip_encoder import CLIPFeatureExtractor
from modules.vector_db import VectorRegistry
from modules.mock_db import MockDatabaseClient

def run_stage4_self_test() -> None:
    """Система smoke-тестирования геометрии (Этап 4)."""
    print("[TEST] Запуск самопроверки модуля пространственной логики...")
    analyzer = SpatialLogicAnalyzer()
    assert analyzer.check_trigger({"boxes": []}, {"landmarks": []}) is False, "Ошибка пустого триггера"
    mock_bad_hand = {"landmarks": [[0.0, 0.0, 0.0]] * 21}
    assert analyzer.check_trigger({"boxes": []}, mock_bad_hand) is False, "Ошибка валидации ложного жеста"
    print("[TEST] Самопроверка успешно пройдена!\n")

def draw_sidebar(dashboard: np.ndarray, fps: float, total_events: int, recent_logs: List[str], trigger_state: str) -> None:
    """
    Отрисовка технологичного UI Сайдбара в правой части экрана (Dashboard).
    Обеспечивает вывод системных метрик, статуса безопасности сцены
    и динамическую ленту логов с цветовым кодированием событий.
    """
    # 1. Заливка фона панели глубоким темно-серым цветом (Carbon/Cyberpunk стиль)
    dashboard[:, :] = (33, 30, 30)  # BGR формат
    
    # Декоративная разделительная неоновая линия между видео и сайдбаром
    cv2.line(dashboard, (0, 0), (0, 720), (55, 53, 50), 2)
    
    # 2. СЕКЦИЯ А: Системные метрики
    cv2.putText(dashboard, "SYSTEM METRICS", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (220, 220, 220), 1)
    cv2.line(dashboard, (20, 55), (380, 55), (75, 71, 68), 1)
    
    # Вывод текущего FPS конвейера
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    
    # Вывод общего счетчика успешных валидаций в БД
    text_log = f"EVENTS LOGGED: {total_events}"
    cv2.putText(dashboard, text_log, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    
    # 3. СЕКЦИЯ Б: Статус Безопасности (Security HUD)
    cv2.putText(dashboard, "SECURITY STATUS:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    
    if trigger_state == "GRANTED":
        # Яркий зеленый прямоугольник при успешном доступе подписчика
        cv2.rectangle(dashboard, (20, 200), (380, 250), (0, 120, 0), -1)
        cv2.putText(dashboard, "ACCESS GRANTED", (65, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
    elif trigger_state == "DENIED":
        # Яркий красный прямоугольник при обнаружении незарегистрированного объекта
        cv2.rectangle(dashboard, (20, 200), (380, 250), (0, 0, 140), -1)
        cv2.putText(dashboard, "ACCESS DENIED", (75, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
    else:
        # Стандартное состояние сканирования сцены (Нейтральный цвет)
        cv2.rectangle(dashboard, (20, 200), (380, 250), (50, 47, 45), -1)
        cv2.putText(dashboard, "SCANNING...", (115, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (160, 160, 160), 1)

    # 4. СЕКЦИЯ В: Живая лента событий (Recent Events Log)
    cv2.putText(dashboard, "RECENT EVENTS LOG", (20, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (220, 220, 220), 1)
    cv2.line(dashboard, (20, 335), (380, 335), (75, 71, 68), 1)
    
    y_offset = 370
    # Итерируемся по последним 5 событиям из истории логов
    for log in recent_logs[-5:]:
        # СЕНИОР-ФИКС: Динамическое цветовое кодирование строк
        # Если в строке лога зафиксирован отказ доступа — подсвечиваем всю строку тревожным КРАСНЫМ цветом
        if "Access Denied" in log:
            log_color = (0, 0, 255)      # Яркий красный (BGR)
        else:
            log_color = (0, 220, 255)    # Неоновый бирюзовый для успешных событий (BGR)
            
        # Отрисовка строки лога (Таймстамп с секундами уже встроен внутри log строки на Этапе инференса)
        cv2.putText(dashboard, log, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, log_color, 1)
        y_offset += 35

def process_frame(frame: np.ndarray, obj_detector: ObjectDetector, hand_detector: HandDetector, 
                  spatial_analyzer: SpatialLogicAnalyzer, clip_encoder: CLIPFeatureExtractor, 
                  vector_db: VectorRegistry, event_db: MockDatabaseClient, snd_success: Any, 
                  snd_denied: Any, log_feed: List[str], trigger_count: int) -> tuple[str, int]:
    """Внутренний конвейер ИИ: Детекция, Трекинг, 3D-Raycasting и CLIP ReID."""
    current_state = "IDLE"
    obj_results = obj_detector.process(frame)
    hand_results = hand_detector.process(frame)
    
    raw_boxes = obj_results.get("boxes")
    boxes = cast(List[List[Any]], raw_boxes if raw_boxes is not None else [])
    raw_landmarks = hand_results.get("landmarks")
    landmarks = cast(List[List[float]], raw_landmarks if raw_landmarks is not None else [])
    
    for box in boxes:
        bx1, by1, bx2, by2, _, btrack_id = box
        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 100, 0), 2)
        cv2.putText(frame, f"ID: {btrack_id}", (int(bx1), int(by1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    
    if landmarks:
        for lm in landmarks:
            cv2.circle(frame, (int(lm[0] * 1280), int(lm[1] * 720)), 4, (0, 255, 0), -1)
            
    is_triggered = spatial_analyzer.check_trigger(obj_results, hand_results)
    if is_triggered:
        target_cup_id = spatial_analyzer.get_last_triggered_id()
        target_box = next((b for b in boxes if len(b) > 5 and b[5] == target_cup_id), None)
        owner_name, reid_score = "Unknown (Unsubscribed)", 0.0
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        
        if target_box is not None:
            x1, y1, x2, y2, _, _ = target_box
            crop = frame[max(0, int(y1)):min(720, int(y2)), max(0, int(x1)):min(1280, int(x2))]
            if crop.size > 0:
                reid_result = vector_db.verify_embedding(clip_encoder.extract(crop))
                owner_name, reid_score = reid_result["name"], reid_result["score"]
                
        time_str = datetime.now().strftime("%H:%M:%S")
        if owner_name != "Unknown (Unsubscribed)":
            current_state = "GRANTED"
            trigger_count += 1
            if snd_success: snd_success.play()
            log_feed.append(f"[{time_str}] {owner_name} - Verified ({reid_score:.2f})")
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), int((x2 - x1) / 2) + 10, (0, 255, 0), 3)
        else:
            current_state = "DENIED"
            if snd_denied: snd_denied.play()
            log_feed.append(f"[{time_str}] Access Denied! ({reid_score:.2f})")
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.line(frame, (int(x1), int(y2)), (int(x2), int(y1)), (0, 0, 255), 3)
            
        event_db.save_event("Trigger_Fired", {"owner": owner_name, "score": reid_score})
        
    return current_state, trigger_count
def main() -> None:
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = config.network.display_env

    print("=== ЗАПУСК МУЛЬТИМОДАЛЬНОЙ СИСТЕМЫ ВЕРИФИКАЦИИ ПОДПИСОК ===")
    try:
        run_stage4_self_test()
    except AssertionError as e:
        print(f"[CRITICAL] Тест логики провален: {e}")
        return

    pygame.mixer.init()
    try:
        snd_success = pygame.mixer.Sound(config.sounds.success_wav)
        snd_denied = pygame.mixer.Sound(config.sounds.denied_wav)
        print("[INFO] Аудиосистема успешно инициализирована.")
    except Exception as e:
        print(f"[WARNING] Не удалось загрузить файлы звука: {e}")
        snd_success, snd_denied = None, None

    obj_detector, hand_detector = ObjectDetector(), HandDetector()
    spatial_analyzer, clip_encoder = SpatialLogicAnalyzer(), CLIPFeatureExtractor()
    vector_db, event_db = VectorRegistry(), MockDatabaseClient()
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
        clip_encoder.load_model()
        event_db.connect()
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return

    os.makedirs(config.video.output_dir, exist_ok=True)
    full_output_path = os.path.join(config.video.output_dir, config.video.output_filename)

    try:
        with VideoStream() as stream:
            video_writer = cv2.VideoWriter(full_output_path, cv2.VideoWriter_fourcc(*'mp4v'), config.video.fps_target, (1680, 720)) # type: ignore
            cv2.namedWindow("Advanced CV - Live AR HUD", cv2.WINDOW_NORMAL)
            
            frame_count, trigger_count = 0, 0
            log_feed: List[str] = ["System started..."]
            prev_time, fps = time.time(), 0.0
            
            for frame in stream.frames():
                frame_count += 1
                if frame_count % 10 == 0:
                    current_time = time.time()
                    fps = 10 / (current_time - prev_time)
                    prev_time = current_time

                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                
                sidebar = np.zeros((720, 400, 3), dtype=np.uint8)
                
                current_state, trigger_count = process_frame(
                    frame, obj_detector, hand_detector, spatial_analyzer, 
                    clip_encoder, vector_db, event_db, snd_success, snd_denied, log_feed, trigger_count
                )
                
                draw_sidebar(sidebar, fps, trigger_count, log_feed, current_state)
                combined_dashboard = np.hstack((frame, sidebar))
                
                cv2.imshow("Advanced CV - Live AR HUD", combined_dashboard)
                video_writer.write(combined_dashboard)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("[INFO] Получен сигнал закрытия окна. Завершение...")
                    break
            
            video_writer.release()
            cv2.destroyAllWindows()
            print(f"[SUCCESS] Результат сохранен в {full_output_path}")
    except Exception as e:
        print(f"[ERROR] Ошибка в главном цикле: {e}")
    finally:
        event_db.disconnect()

if __name__ == "__main__":
    main()
