from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime

from src.models.database import Conversation

class SearchService:
    def __init__(self, db: Session):
        self.db = db
    
    def search_conversations(
        self,
        user_id: int,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        phase: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> List[Dict]:
        """
        Search conversations with various filters
        """
        query_obj = self.db.query(Conversation).filter(Conversation.user_id == user_id)
        
        if query:
            query_obj = query_obj.filter(
                or_(
                    Conversation.transcript["text"].as_string().ilike(f"%{query}%"),
                    Conversation.analysis["segments"].as_string().ilike(f"%{query}%")
                )
            )
        
        if start_date:
            query_obj = query_obj.filter(Conversation.created_at >= start_date)
        
        if end_date:
            query_obj = query_obj.filter(Conversation.created_at <= end_date)
        
        if phase:
            query_obj = query_obj.filter(
                Conversation.analysis["summary"]["phase_distribution"][phase].as_numeric() > 0
            )
        
        if sentiment:
            query_obj = query_obj.filter(
                Conversation.analysis["summary"]["sentiment_summary"][sentiment].as_numeric() > 0
            )
        
        return [conversation.to_dict() for conversation in query_obj.all()]

    def get_conversation_stats(self, user_id: int) -> Dict:
        """
        Get aggregated statistics for user's conversations
        """
        conversations = self.db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).all()
        
        total_duration = 0
        phase_stats = {
            "introduction": 0,
            "discovery": 0,
            "pitch": 0,
            "objection_handling": 0,
            "closing": 0
        }
        sentiment_stats = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        
        for conv in conversations:
            total_duration += conv.analysis["summary"]["duration"]
            
            # Aggregate phase stats
            for phase, duration in conv.analysis["summary"]["phase_distribution"].items():
                phase_stats[phase] += duration
            
            # Aggregate sentiment stats
            for sentiment, count in conv.analysis["summary"]["sentiment_summary"].items():
                sentiment_stats[sentiment] += count
        
        return {
            "total_conversations": len(conversations),
            "total_duration": total_duration,
            "phase_distribution": phase_stats,
            "sentiment_distribution": sentiment_stats
        }