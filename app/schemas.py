from pydantic import BaseModel
from typing import Dict, List

class MatchInput(BaseModel):
    home_team: str
    away_team: str
    match_date: str

class PredictionOutput(BaseModel):
    predicted_outcome: str
    probabilities: Dict[str, float]