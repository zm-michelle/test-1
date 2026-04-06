import os
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import json
load_dotenv()


class LLMConfiguration(BaseModel):
    """Configuration for the agent"""

    smart_llm: str = Field(
        default="ollama:qwen2.5:7b-instruct",
        description="model to use for rewriting"
    )
    fast_llm: str = Field(
        default="ollama:llama3.2",
        description="model to use for fast tasks"
    )
    ollama_endpoints: list[str] = Field(
        default=["http://localhost:11434"]
    )
    temperature: float = Field(default=0.3, description="LLM Temperature")
    max_workers: Optional[int] = Field(
        default=None,
        description="Cap parallelism. None = use get_optimal_workers()"
    )
    max_tokens: int = Field(default=5000, description="Max tokens for LLM output")
    max_rewrite_attempts: int = Field(
        default=3,
        description="How many times to retry a rewrite if quality check fails"
    )

    def _is_ollama(self, model: str) -> bool:
        return model.startswith("ollama:")

    def get_smart_llm(self, endpoint_index: int = 0) -> BaseChatModel:
        if self._is_ollama(self.smart_llm):
            return ChatOllama(
                model=self.smart_llm.split(":", 1)[1],
                temperature=self.temperature,
                base_url=self.ollama_endpoints[endpoint_index],
            )
        return init_chat_model(self.smart_llm, temperature=self.temperature)

    def get_fast_llm(self) -> BaseChatModel:
        if self._is_ollama(self.fast_llm):
            return ChatOllama(
                model=self.fast_llm.split(":", 1)[1],
                temperature=self.temperature,
                base_url=self.ollama_endpoints[0],
            )
        return init_chat_model(self.fast_llm, temperature=self.temperature)

    def is_local(self, model) -> bool:
        return not any(
            model.startswith(prefix)
            for prefix in ("claude", "gpt", "openai", "anthropic")
        )

    def get_optimal_workers(self) -> int:
        if self._is_ollama(self.smart_llm):
            return len(self.ollama_endpoints)
        elif self.max_workers is not None:
            return self.max_workers
        return -1

    

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "LLMConfiguration":
        configurable = config.get("configurable", {}) if config else {}
        values: dict[str, Any] = {}
        for field in cls.model_fields:
            val = os.environ.get(field.upper(), configurable.get(field))
            if val is None:
                continue
            # parse JSON strings for list fields
            if isinstance(val, str) and val.startswith("["):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            values[field] = val
        return cls(**values)

class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore", env_file=".env")

    session_ttl_seconds: int = 86400
    session_prefix: str = "session:"
    redis_url: str = "redis://localhost:6375/0"
    postgres_dns: str = "postgresql://"


settings = Settings()