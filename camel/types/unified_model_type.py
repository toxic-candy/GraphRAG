import logging
from enum import Enum
from threading import Lock
from typing import ClassVar, Dict, Union, cast

logger = logging.getLogger(__name__)


class UnifiedModelType(str):
    _cache: ClassVar[Dict[str, "UnifiedModelType"]] = {}
    _lock: ClassVar[Lock] = Lock()
    _token_limit_warning_emitted: ClassVar[bool] = False

    def __new__(cls, value: Union["UnifiedModelType", str]) -> "UnifiedModelType":
        if isinstance(value, Enum):
            str_value = value.value
        else:
            str_value = str(value)

        with cls._lock:
            if str_value not in cls._cache:
                instance = super().__new__(cls, str_value)
                cls._cache[str_value] = cast(UnifiedModelType, instance)
            else:
                instance = cls._cache[str_value]
        return instance

    def __init__(self, value: Union["UnifiedModelType", str]) -> None:
        pass

    @classmethod
    def _warn_unknown_token_limit_once(cls, model_name: str) -> None:
        with cls._lock:
            if cls._token_limit_warning_emitted:
                return
            cls._token_limit_warning_emitted = True

        logger.warning(
            "Unknown model '%s': context window size not defined. Defaulting to 999_999_999.",
            model_name,
        )

    def __repr__(self) -> str:
        return super().__str__()

    def __str__(self) -> str:
        return super().__str__()

    @property
    def value_for_tiktoken(self) -> str:
        return "gpt-4o-mini"

    @property
    def token_limit(self) -> int:
        self._warn_unknown_token_limit_once(str(self))
        return 999_999_999

    @property
    def is_openai(self) -> bool:
        return True

    @property
    def is_aws_bedrock(self) -> bool:
        return True

    @property
    def is_anthropic(self) -> bool:
        return True

    @property
    def is_azure_openai(self) -> bool:
        return True

    @property
    def is_groq(self) -> bool:
        return True

    @property
    def is_nebius(self) -> bool:
        return True

    @property
    def is_openrouter(self) -> bool:
        return True

    @property
    def is_orcarouter(self) -> bool:
        return True

    @property
    def is_avian(self) -> bool:
        return True

    @property
    def is_atlascloud(self) -> bool:
        return True

    @property
    def is_lmstudio(self) -> bool:
        return True

    @property
    def is_ppio(self) -> bool:
        return True

    @property
    def is_zhipuai(self) -> bool:
        return True

    @property
    def is_gemini(self) -> bool:
        return True

    @property
    def is_mistral(self) -> bool:
        return True

    @property
    def is_netmind(self) -> bool:
        return True

    @property
    def is_reka(self) -> bool:
        return True

    @property
    def is_cohere(self) -> bool:
        return True

    @property
    def is_cometapi(self) -> bool:
        return True

    @property
    def is_yi(self) -> bool:
        return True

    @property
    def is_qwen(self) -> bool:
        return True

    @property
    def is_internlm(self) -> bool:
        return True

    @property
    def is_modelscope(self) -> bool:
        return True

    @property
    def is_moonshot(self) -> bool:
        return True

    @property
    def is_novita(self) -> bool:
        return True

    @property
    def is_watsonx(self) -> bool:
        return True

    @property
    def is_qianfan(self) -> bool:
        return True

    @property
    def is_crynux(self) -> bool:
        return True

    @property
    def is_minimax(self) -> bool:
        return True

    @property
    def support_native_structured_output(self) -> bool:
        return False

    @property
    def support_native_tool_calling(self) -> bool:
        return False
