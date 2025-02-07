from typing import Dict
from datetime import datetime
import asyncio
from sqlalchemy.orm import Session
import json

class ProcessingProgress:
    def __init__(self):
        self.tasks = {}
    
    def create_task(self, task_id: str, total_steps: int = 4) -> None:
        """
        Initialize a new processing task
        """
        self.tasks[task_id] = {
            "started_at": datetime.utcnow(),
            "current_step": 0,
            "total_steps": total_steps,
            "status": "processing",
            "step_details": {
                0: "Uploading file",
                1: "Converting audio",
                2: "Transcribing",
                3: "Analyzing"
            },
            "error": None
        }
    
    def update_progress(self, task_id: str, step: int, status: str = "processing", error: str = None) -> None:
        """
        Update the progress of a task
        """
        if task_id in self.tasks:
            self.tasks[task_id].update({
                "current_step": step,
                "status": status,
                "error": error
            })
    
    def get_progress(self, task_id: str) -> Dict:
        """
        Get the current progress of a task
        """
        if task_id not in self.tasks:
            return {"error": "Task not found"}
        
        task = self.tasks[task_id]
        progress = (task["current_step"] / task["total_steps"]) * 100
        
        return {
            "task_id": task_id,
            "progress": progress,
            "current_step": task["current_step"],
            "step_details": task["step_details"].get(task["current_step"]),
            "status": task["status"],
            "error": task["error"],
            "started_at": task["started_at"].isoformat()
        }
    
    def clean_old_tasks(self, max_age_hours: int = 24) -> None:
        """
        Remove completed tasks older than specified hours
        """
        current_time = datetime.utcnow()
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if task["status"] in ["completed", "error"]:
                age = (current_time - task["started_at"]).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]

# Global progress tracker instance
progress_tracker = ProcessingProgress()

# Background task to clean old tasks periodically
async def clean_old_tasks_periodically():
    while True:
        progress_tracker.clean_old_tasks()
        await asyncio.sleep(3600)  # Clean every hour