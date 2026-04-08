from typing import List

from pydantic import BaseModel, Field


class EmotionResult(BaseModel):
    """Structured analysis result returned by the LLM."""

    emotion: str = Field(..., description="positive | neutral | negative")
    risk_level: str = Field(..., description="low | medium | high")
    depression_score: float = Field(0.0, description="Estimated depression score 0.0..1.0")
    anxiety_score: float = Field(0.0, description="Estimated anxiety score 0.0..1.0")
    stress_score: float = Field(0.0, description="Estimated stress score 0.0..1.0")
    summary: str = Field(..., description="Summary for specialist dashboard")
    profile_note: str = Field("", description="Short note about tone/needs/resources")
    user_reply: str = Field("", description="Empathetic assistant reply for the patient")
    key_characteristics: List[str] = Field(
        default_factory=list,
        description="Short patient characteristics that may help a specialist",
    )
