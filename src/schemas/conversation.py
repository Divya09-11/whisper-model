from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class ConversationBase(BaseModel):
    file_path: str
    transcript: Optional[Dict] = None
    analysis: Optional[Dict] = None

class ConversationCreate(ConversationBase):
    user_id: int

class ConversationResponse(ConversationBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True