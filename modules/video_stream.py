import cv2
import socket
import pickle
import struct
import numpy as np
from typing import Generator, Any
from config import config

class VideoStream:
    """
    Универсальный класс видеопотока (Enterprise-уровень).
    Поддерживает как стандартные файлы/камеры OpenCV, так и чтение живых кадров 
    из сетевого TCP-моста для работы в изолированных средах WSL2.
    """
    def __init__(self) -> None:
        self._source: str | int = config.video.source
        self._width: int = config.video.width
        self._height: int = config.video.height
        
        # Дескрипторы для двух разных режимов работы
        self._cap: cv2.VideoCapture | None = None
        self._client_socket: socket.socket | None = None

    def __enter__(self) -> "VideoStream":
        """Инициализирует источник видео (локальный или сетевой) в зависимости от конфига."""
        if self._source == "wsl_network":
            # Режим сетевого моста: подключаемся к скрипту NetStream.py на виртуальной Windows 11
            self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        
            host_ip = "192.168.182.20" 
            port = 9999
            
            print(f"[WSL NETWORK] Попытка подключения к камере виртуальной Windows 11 по IP: {host_ip}:{port}")
            try:
                self._client_socket.connect((host_ip, port))
                print("[WSL NETWORK] Сетевой мост успешно установлен!")
            except Exception as e:
                raise RuntimeError(f"Не удалось подключиться к сетевому стриму камеры: {e}")
        else:
            # Стандартный режим: чтение файлов или локального индекса 0
            self._cap = cv2.VideoCapture(self._source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Не удалось открыть источник видео: {self._source}")
            
            if isinstance(self._source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                
        return self

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Генератор, лениво возвращающий кадры из выбранного источника."""
        # Режим А: Чтение сырых байт из сокета и десериализация матриц пикселей в numpy массив
        if self._source == "wsl_network" and self._client_socket:
            data = b""
            payload_size = struct.calcsize("Q")
            while True:
                try:
                    # Читаем размер заголовка кадра
                    while len(data) < payload_size:
                        packet = self._client_socket.recv(4 * 1024)
                        if not packet: 
                            break
                        data += packet
                    if len(data) < payload_size:
                        break
                        
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]
                    
                    # Читаем само тело кадра (сериализованный массив пикселей)
                    while len(data) < msg_size:
                        data += self._client_socket.recv(4 * 1024)
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    
                    # Восстанавливаем кадр OpenCV
                    frame = pickle.loads(frame_data)
                    yield frame
                except Exception:
                    print("[WSL NETWORK] Соединение с сервером камеры разорвано или завершено.")
                    break
                    
        # Режим Б: Стандартное покадровое чтение файла методами OpenCV
        elif self._cap:
            while self._cap.isOpened():
                success, frame = self._cap.read()
                if not success:
                    break
                yield frame

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Безопасно закрывает дескрипторы ресурсов при выходе из контекста 'with'."""
        if self._cap is not None:
            self._cap.release()
        if self._client_socket:
            self._client_socket.close()
            print("[WSL NETWORK] Сетевой сокет закрыт.")
