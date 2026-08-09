from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime

class IDatabaseClient(ABC):
    """
    Абстрактный интерфейс для сохранения зафиксированных CV-событий.
    Обеспечивает независимость бизнес-логики от конкретной реализации хранилища данных.
    """

    @abstractmethod
    def connect(self) -> None:
        """Устанавливает соединение с хранилищем данных / открывает дескриптор файла."""
        pass

    @abstractmethod
    def save_event(self, event_type: str, metadata: Dict[str, Any]) -> bool:
        """
        Регистрирует событие в системе.
        
        :param event_type: Строковый идентификатор типа события (например, 'Jumbo_Over_Cup')
        :param metadata: Дополнительные контекстные данные (координаты, уверенность нейросети)
        :return: True, если запись прошла успешно, иначе False
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Безопасно закрывает соединение / сохраняет буферы на диск."""
        pass
