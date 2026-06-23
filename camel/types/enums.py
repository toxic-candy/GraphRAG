import os
from enum import Enum, EnumMeta


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    CRITIC = "critic"
    EMBODIMENT = "embodiment"
    DEFAULT = "default"


class TaskType(Enum):
    AI_SOCIETY = "ai_society"
    CODE = "code"
    MISALIGNMENT = "misalignment"
    TRANSLATION = "translation"
    EVALUATION = "evaluation"
    SOLUTION_EXTRACTION = "solution_extraction"
    ROLE_DESCRIPTION = "role_description"
    GENERATE_TEXT_EMBEDDING_DATA = "generate_text_embedding_data"
    OBJECT_RECOGNITION = "object_recognition"
    IMAGE_CRAFT = "image_craft"
    MULTI_CONDITION_IMAGE_CRAFT = "multi_condition_image_craft"
    DEFAULT = "default"
    VIDEO_DESCRIPTION = "video_description"


class VectorDistance(Enum):
    DOT = "dot"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class OpenAIBackendRole(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    FUNCTION = "function"
    TOOL = "tool"


class OpenAIImageTypeMeta(EnumMeta):
    def __contains__(cls, image_type: object) -> bool:
        try:
            cls(image_type)
        except ValueError:
            return False
        return True


class OpenAIImageType(Enum, metaclass=OpenAIImageTypeMeta):
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    GIF = "gif"


class OpenAIVisionDetailType(Enum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class JinaReturnFormat(Enum):
    DEFAULT = None
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class ModelType(Enum):
    DEFAULT = os.getenv("DEFAULT_MODEL_TYPE", "gpt-4.1-mini-2025-04-14")
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"
    O1 = "o1"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"
    GPT_5 = "gpt-5"
    GPT_5_1 = "gpt-5.1"
    GPT_5_2 = "gpt-5.2"
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini-2026-03-17"
    GPT_5_4_NANO = "gpt-5.4-nano-2026-03-17"
    GPT_5_4_PRO = "gpt-5.5-pro-2026-04-23"
    GPT_5_5 = "gpt-5.5"
    GPT_5_5_PRO = "gpt-5.5-pro"
    LLAMA_2 = "llama-2"
    LLAMA_3 = "llama-3"
    GROQ_LLAMA_3_8B = "groq-llama-3-8b"
    GROQ_LLAMA_3_70B = "groq-llama-3-70b"
    VICUNA = "vicuna"
    VICUNA_16K = "vicuna-16k"
    GLM_4_OPEN_SOURCE = "glm-4-open-source"
    QWEN_2 = "qwen-2"
    GROQ_MIXTRAL_8_7B = "groq-mixtral-8x7b"
    GROQ_GEMMA_7B_IT = "groq-gemma-7b-it"
    GROQ_GEMMA_2_9B_IT = "groq-gemma-2-9b-it"
    MISTRAL_CODESTRAL = "mistral-codestral"
    MISTRAL_CODESTRAL_MAMBA = "mistral-codestral-mamba"
