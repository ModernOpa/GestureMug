import json
from datetime import datetime
from typing import Any, Dict, IO
from interfaces.database import IDatabaseClient
from config import config

class MockDatabaseClient(IDatabaseClient):
    """
    Реализация клиента базы данных (Mock). 
    Записывает зафиксированные CV-события в текстовый лог-файл в формате JSON-строк.
    """
    def __init__(self) -> None:
        self._log_path: str = config.log_file_path
        self._file: IO[str] | None = None

    def connect(self) -> None:
        """Открывает дескриптор файла в режиме добавления записей (Append)."""
        try:
            self._file = open(self._log_path, mode="a", encoding="utf-8")
            print(f"[INFO] Установлено соединение с базой данных (файл: {self._log_path})")
        except IOError as e:
            raise RuntimeError(f"Не удалось инициализировать хранилище данных: {e}")

    def save_event(self, event_type: str, metadata: Dict[str, Any]) -> bool:
        """
        Сериализует событие в JSON и записывает в файл.
        """
        if self._file is None:
            print("[ERROR] Попытка записи в закрытую базу данных.")
            return False

        # Формируем документ события по канонам баз данных
        event_document = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "metadata": metadata
        }

        try:
            # Записываем как структурированную JSON-строку
            self._file.write(json.dumps(event_document, ensure_ascii=False) + "\n")
            self._file.flush()  # Принудительно сбрасываем буфер на диск
            return True
        except IOError:
            print(f"[ERROR] Ошибка записи события {event_type} на диск.")
            return False

    def disconnect(self) -> None:
        """Безопасно закрывает файл."""
        if self._file and not self._file.closed:
            self._file.close()
            print("[INFO] Соединение с базой данных успешно закрыто.")
