from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from pydantic import BaseModel


class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str, config: Dict[str, Any]) -> Any:
        """Invokes the language model with a given prompt and configuration."""
        pass

    @abstractmethod
    def invoke_structured(
        self, prompt: str, schema: Type[BaseModel], config: Dict[str, Any] = None
    ) -> BaseModel:
        """Invokes the model and returns a `schema` instance instead of raw text."""
        pass
