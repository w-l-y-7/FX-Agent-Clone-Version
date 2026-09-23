from typing import Any, Dict, Type

from pydantic import BaseModel

from ..core.abstractions.base_llm import BaseLLM

class MockLLMService(BaseLLM):
    """A mock LLM service for development and testing."""
    def invoke(self, prompt: str, config: Dict[str, Any] = None) -> str:
        print(f"--- MockLLMService Invoked ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Config: {config}")
        return "This is a mock plan from MockLLMService."

    def invoke_structured(
        self, prompt: str, schema: Type[BaseModel], config: Dict[str, Any] = None
    ) -> BaseModel:
        print(f"--- MockLLMService Invoked (structured: {schema.__name__}) ---")
        return schema(
            market_commentary="This is a mock market commentary.",
            features_for_forecasting=["feature1", "feature2"],
        )