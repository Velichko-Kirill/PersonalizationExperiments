from typing import Dict, List, Union
from langchain_core.pydantic_v1 import BaseModel, Field


class History(BaseModel):
    event_id: int = Field(description="Unique event_id")
    date: str = Field(description="data of event in format mm.dd.yyyy")
    event: str = Field(description="The short description of the event")
    feature: str = Field(
        description="Feature to be changed as a result of action")
    new_value: str = Field(description="New value of the feature changed")


class SummaryRequest(BaseModel):
    text: str
    num_words: int
