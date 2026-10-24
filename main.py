import cv2
import os
import time
import socket
import subprocess
from datetime import datetime
from typing import List, Dict, Any, cast, Optional
import numpy as np

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
    
    # Сценарий 1: Пустые данные
    assert analyzer.check_trigger({"boxes": []}, {"landmarks": []}) is False, "Ошибка пустого триггера"
    
    # Сценарий 2: Неверный жест (синтетическая пустая ладонь)
    mock_bad_hand = {"landmarks": [[0.0, 0.0, 0.0]] * 21}
    assert analyzer.check_trigger({"boxes": []}, mock_bad_hand) is False, "Ошибка валидации ложного жеста"
    
    print("[TEST] Самопроверка успешно пройдена!\n")

def draw_sidebar(dashboard: np.ndarray, fps: float, total_events: int, recent_logs: List[str], trigger_state: str) -> None:
    """Отрисовка технологичного UI Сайдбара в правой части экрана (Dashboard)."""
    # Заливка фонда панели глубоким темно-серым цветом
    dashboard[:, :] = (33, 30, 30)
    
    # Декоративная разделительная неоновая линия между видео и сайдбаром
    cv2.line(dashboard, (0, 0), (0, config.video.height), (55, 53, 50), 2)
    
    # СЕКЦИЯ А: Системные метрики
    cv2.putText(dashboard, "SYSTEM METRICS", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (220, 220, 220), 1)
    cv2.line(dashboard, (20, 55), (380, 55), (75, 71, 68), 1)
    
    # Вывод текущего FPS конвейера
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    
    # Вывод общего счетчика успешных валидаций в БД
    text_log = f"EVENTS LOGGED: {total_events}"
    cv2.putText(dashboard, text_log, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    
    # СЕКЦИЯ Б: Статус Безопасности (Security HUD)
    cv2.putText(dashboard, "SECURITY STATUS:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    
    if trigger_state == "GRANTED":
        cv2.rectangle(dashboard, (20, 200), (380, 250), (0, 120, 0), -1)
        cv2.putText(dashboard, "ACCESS GRANTED", (65, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
    elif trigger_state == "DENIED":
        cv2.rectangle(dashboard, (20, 200), (380, 250), (0, 0, 140), -1)
        cv2.putText(dashboard, "ACCESS DENIED", (75, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.rectangle(dashboard, (20, 200), (380, 250), (50, 47, 45), -1)
        cv2.putText(dashboard, "SCANNING...", (115, 235), cv2.FONT_HERSHEY_DUPLEX, 0.65, (160, 160, 160), 1)

    # СЕКЦИЯ В: Живая лента событий (Recent Events Log)
    cv2.putText(dashboard, "RECENT EVENTS LOG", (20, 320), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (220, 220, 220), 1)
    cv2.line(dashboard, (20, 335), (380, 335), (75, 71, 68), 1)
    
    y_offset = 370
    for log in recent_logs[-5:]:
        if "Access Denied" in log:
            log_color = (0, 0, 255)
        else:
            log_color = (0, 220, 255)
            
        cv2.putText(dashboard, log, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, log_color, 1)
        y_offset += 35
def process_frame(frame: np.ndarray, obj_detector: ObjectDetector, hand_detector: HandDetector, 
                  spatial_analyzer: SpatialLogicAnalyzer, clip_encoder: CLIPFeatureExtractor, 
                  vector_db: VectorRegistry, event_db: MockDatabaseClient, sound_socket: Optional[socket.socket], 
                  log_feed: List[str], trigger_count: int) -> tuple[str, int]:
    """Внутренний конвейер ИИ: Детекция, Трекинг, 3D-Raycasting и CLIP ReID."""
    current_state = "IDLE"
    obj_results = obj_detector.process(frame)
    hand_results = hand_detector.process(frame)
    
    raw_boxes = obj_results.get("boxes")
    boxes = cast(List[List[Any]], raw_boxes if raw_boxes is not None else [])
    
    # 1. Безопасный цикл отрисовки кружек YOLO
    for box in boxes:
        bx1, by1, bx2, by2, _, btrack_id = box
        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 100, 0), 2)
        cv2.putText(frame, f"ID: {btrack_id}", (int(bx1), int(by1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    
    # 2. СТРОГИЙ СЕНИОР-ФИКС: Корректная распаковка и отрисовка суставов руки MediaPipe 1.x Tasks API
    # В новом API результат — это объект, содержащий hand_landmarks как список списков структур
    if "landmarks" in hand_results and hand_results["landmarks"]:
        for landmark_list in hand_results["landmarks"]:
            # Проверяем, если нам пришел плоский список координат [x, y, z] от нашего модуля
            if isinstance(landmark_list, list) and len(landmark_list) > 0:
                # Если это вложенный список точек [[x,y,z], [x,y,z]...]
                if isinstance(landmark_list[0], list):
                    for lm in landmark_list:
                        pt_x = int(lm[0] * config.video.width)
                        pt_y = int(lm[1] * config.video.height)
                        cv2.circle(frame, (pt_x, pt_y), 5, (0, 255, 0), -1)
                else:
                    # Если пришел плоский массив координат одной точки
                    pt_x = int(landmark_list[0] * config.video.width)
                    pt_y = int(landmark_list[1] * config.video.height)
                    cv2.circle(frame, (pt_x, pt_y), 5, (0, 255, 0), -1)
            
    is_triggered = spatial_analyzer.check_trigger(obj_results, hand_results)
    if is_triggered:
        target_cup_id = spatial_analyzer.get_last_triggered_id()
        target_box = next((b for b in boxes if len(b) > 5 and b[5] == target_cup_id), None)
        owner_name, reid_score = "Unknown (Unsubscribed)", 0.0
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        
        if target_box is not None:
            x1, y1, x2, y2, _, _ = target_box
            crop = frame[max(0, int(y1)):min(config.video.height, int(y2)), max(0, int(x1)):min(config.video.width, int(x2))]
            if crop.size > 0:
                reid_result = vector_db.verify_embedding(clip_encoder.extract(crop))
                owner_name, reid_score = reid_result["name"], reid_result["score"]
                
        time_str = datetime.now().strftime("%H:%M:%S")
        if owner_name != "Unknown (Unsubscribed)":
            current_state = "GRANTED"
            trigger_count += 1
            if sound_socket:
                try:
                    sound_socket.sendall(b"SUCCESS")
                except socket.error:
                    print("[WARNING] Сбой отправки аудио-команды SUCCESS")
            log_feed.append(f"[{time_str}] {owner_name} - Verified ({reid_score:.2f})")
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), int((x2 - x1) / 2) + 10, (0, 255, 0), 3)
        else:
            current_state = "DENIED"
            if sound_socket:
                try:
                    sound_socket.sendall(b"DENIED")
                except socket.error:
                    print("[WARNING] Сбой отправки аудио-команды DENIED")
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

    # Индустриальная инициализация сетевого звукового сокета (Считывание IP и Порта строго из config)
    sound_socket: Optional[socket.socket] = None
    try:
        # СЕНИОР-ФИКС: Берем проверенный Host-Only IP виртуалки из конфигурации, а не шлюз WSL!
        host_ip = config.network.host_ip
        sound_port = int(getattr(config.network, "sound_port", 9996))
        
        sound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Устанавливаем таймаут в 3 секунды, чтобы скрипт не висел в случае блокировок
        sound_socket.settimeout(3.0) 
        
        print(f"[WSL NETWORK] Попытка подключения к аудио-серверу VMware по IP: {host_ip}:{sound_port}")
        sound_socket.connect((host_ip, sound_port))
        
        # Возвращаем сокет в блокирующий режим после успешного коннекта
        sound_socket.settimeout(None) 
        print(f"[WSL NETWORK] Аудио-мост с Windows Sound Server успешно установлен на порту {sound_port}!")
    except Exception as e:
        print(f"[WARNING] Не удалось подключить Windows аудио-сервер: {e}. Сцена без живого звука.")
        sound_socket = None


    obj_detector, hand_detector = ObjectDetector(), HandDetector()
    spatial_analyzer, clip_encoder = SpatialLogicAnalyzer(), CLIPFeatureExtractor()
    vector_db, event_db = VectorRegistry(), MockDatabaseClient()
    
    try:
        obj_detector.load_model()
        hand_detector.load_model()
        clip_encoder.load_model()
        event_db.connect()
    except Exception as e:
        print(f"Ошибка инициализации моделей: {e}")
        if sound_socket:
            sound_socket.close()
        return

    os.makedirs(config.video.output_dir, exist_ok=True)
    full_output_path = os.path.join(config.video.output_dir, config.video.output_filename)

    try:
        with VideoStream() as stream:
            total_width = config.video.width + 400
            video_writer = cv2.VideoWriter(
                full_output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
                config.video.fps_target, 
                (total_width, config.video.height)
            )
            
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

                if frame.shape[1] != config.video.width or frame.shape[0] != config.video.height:
                    frame = cv2.resize(frame, (config.video.width, config.video.height))
                
                sidebar = np.zeros((config.video.height, 400, 3), dtype=np.uint8)
                
                current_state, trigger_count = process_frame(
                    frame, obj_detector, hand_detector, spatial_analyzer, 
                    clip_encoder, vector_db, event_db, sound_socket, log_feed, trigger_count
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
        if sound_socket:
            sound_socket.close()
        event_db.disconnect()

if __name__ == "__main__":
    main()
