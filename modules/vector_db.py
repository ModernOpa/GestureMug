import json
import os
from typing import Any, Dict, List

import numpy as np

from config import config


class VectorRegistry:
    """
    In-Memory Хранилище векторных эмбеддингов кружек (База Данных Подписчиков).
    Использует Косинусное Сходство (Cosine Similarity) для поиска совпадений.
    """
    def __init__(self) -> None:        
        self._db_path: str = config.vector_db.db_path
        self._reid_threshold: float = config.vector_db.reid_threshold
        
        self._registry: Dict[str, List[float]] = {}
        self.load_db()

    def load_db(self) -> None:
        """Загружает базу векторов из файла JSON, если он существует."""
        if os.path.exists(self._db_path):
            try:
                with open(self._db_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
                print(f"[VECTOR DB] База данных успешно загружена. Объектов в индексе: {len(self._registry)}")
            except Exception as e:
                print(f"[VECTOR DB] Ошибка при чтении базы данных, создана пустая: {e}")
                self._registry = {}
        else:
            dir_name = os.path.dirname(self._db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self.save_db()

    def save_db(self) -> None:
        """Сохраняет текущую базу векторов на диск."""
        with open(self._db_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)

    def add_subscriber(self, name: str, embedding: np.ndarray) -> None:
        """Добавляет новый эмбеддинг кружки в базу 'своих' объектов."""
        self._registry[name] = embedding.tolist()
        self.save_db()
        print(f"[VECTOR DB] Объект '{name}' успешно зарегистрирован в базе.")

    def delete_subscriber(self, name: str) -> bool:
        """Удаляет кружку из списка 'своих'."""
        if name in self._registry:
            del self._registry[name]
            self.save_db()
            return True
        return False

    def verify_embedding(self, target_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Ищет наиболее похожий эмбеддинг в базе методом Cosine Similarity.
        """
        if not self._registry:
            return {"verified": False, "name": "Unknown", "score": 0.0}

        best_score = -1.0
        best_name = "Unknown"

        norm_target = target_embedding / np.linalg.norm(target_embedding)

        for name, vec_list in self._registry.items():
            vec = np.array(vec_list)
            norm_vec = vec / np.linalg.norm(vec)
            
            similarity = float(np.dot(norm_target, norm_vec))
            
            if similarity > best_score:
                best_score = similarity
                best_name = name

        # Используем порог из конфигурации
        is_verified = best_score >= self._reid_threshold

        return {
            "verified": is_verified,
            "name": best_name if is_verified else "Unknown (Unsubscribed)",
            "score": round(best_score, 4)
        }
