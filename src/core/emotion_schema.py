from pydantic import BaseModel, Field

class EmotionResult(BaseModel):
    """Валидация JSON-ответа от модели."""
    emotion: str = Field(..., description="positive / neutral / negative")
    risk_level: str = Field(..., description="low / medium / high")
    summary: str = Field(..., description="краткое пояснение вывода модели")
