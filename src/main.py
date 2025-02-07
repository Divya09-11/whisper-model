from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, status, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import uuid

from .models.database import get_db, Conversation
from src.services.auth import get_current_user
from .services.audio_processing import process_audio_file
from .services.transcription import transcribe_audio
from .services.classification import classify_dialogue
from .services.speaker_identification import SpeakerIdentifier
from .services.progress import progress_tracker
from .services.search import SearchService
from .schemas.conversation import ConversationCreate, ConversationResponse

app = FastAPI(title="Sales Conversation Analysis System")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speaker_identifier = SpeakerIdentifier()

@app.post("/upload/single", response_model=ConversationResponse)
async def upload_single_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process a single audio file for transcription and analysis.
    """
    # Validate file
    if file.content_type not in ["audio/wav", "audio/mp3", "audio/m4a"]:
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Generate task ID for progress tracking
    task_id = str(uuid.uuid4())
    progress_tracker.create_task(task_id)
    
    try:
        # Process file
        progress_tracker.update_progress(task_id, 0)
        file_path = await process_audio_file(file)
        
        # Add to processing queue
        background_tasks.add_task(
            transcribe_and_analyze,
            file_path,
            db,
            current_user.id,
            task_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "File uploaded successfully, processing started",
                "task_id": task_id
            }
        )
    except Exception as e:
        progress_tracker.update_progress(task_id, 0, "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/batch", response_model=list[ConversationResponse])
async def upload_batch_audio(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process multiple audio files for transcription and analysis.
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    task_ids = []
    for file in files:
        if file.content_type not in ["audio/wav", "audio/mp3", "audio/m4a"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format for {file.filename}"
            )
    
    responses = []
    for file in files:
        task_id = str(uuid.uuid4())
        progress_tracker.create_task(task_id)
        
        try:
            file_path = await process_audio_file(file)
            background_tasks.add_task(
                transcribe_and_analyze,
                file_path,
                db,
                current_user.id,
                task_id
            )
            
            responses.append({
                "message": f"File {file.filename} uploaded successfully",
                "task_id": task_id
            })
            task_ids.append(task_id)
        except Exception as e:
            progress_tracker.update_progress(task_id, 0, "error", str(e))
            responses.append({
                "message": f"Error processing {file.filename}: {str(e)}",
                "task_id": task_id,
                "error": str(e)
            })
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=responses
    )

@app.get("/progress/{task_id}")
async def get_task_progress(
    task_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get the progress of a specific task.
    """
    progress = progress_tracker.get_progress(task_id)
    if "error" in progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress

@app.get("/conversations/search")
async def search_conversations(
    query: str = None,
    start_date: str = None,
    end_date: str = None,
    phase: str = None,
    sentiment: str = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search conversations with various filters.
    """
    search_service = SearchService(db)
    return search_service.search_conversations(
        user_id=current_user.id,
        query=query,
        start_date=start_date,
        end_date=end_date,
        phase=phase,
        sentiment=sentiment
    )

@app.get("/conversations/stats")
async def get_conversation_stats(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get aggregated statistics for user's conversations.
    """
    search_service = SearchService(db)
    return search_service.get_conversation_stats(current_user.id)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Conversation Analysis System!"}

@app.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: int,
    format: str = "json",
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export a conversation in the specified format (json, csv, or txt)
    """
    # Verify conversation belongs to user
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if format not in ["json", "csv", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    if format == "json":
        return JSONResponse(
            content={
                "transcript": conversation.transcript,
                "analysis": conversation.analysis,
                "metadata": {
                    "created_at": conversation.created_at.isoformat(),
                    "file_path": conversation.file_path
                }
            }
        )
    elif format == "csv":
        csv_content = "timestamp,speaker,text,phase,sentiment\n"
        for segment in conversation.transcript["segments"]:
            csv_content += f"{segment['start']:.2f},{segment['speaker']},\"{segment['text']}\","
            classification = next(s["classification"] for s in conversation.analysis["segments"] 
                              if s["start"] == segment["start"])
            csv_content += f"{classification['phase']},{classification['sentiment']}\n"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.csv"
            }
        )
    else:  # txt format
        txt_content = f"Sales Conversation Analysis - ID: {conversation_id}\n"
        txt_content += f"Date: {conversation.created_at}\n\n"
        txt_content += "Transcript:\n"
        for segment in conversation.transcript["segments"]:
            txt_content += f"[{segment['start']:.2f}s] {segment['speaker']}: {segment['text']}\n"
        
        txt_content += "\nAnalysis:\n"
        txt_content += f"Duration: {conversation.analysis['summary']['duration']:.2f}s\n"
        txt_content += "\nPhase Distribution:\n"
        for phase, duration in conversation.analysis["summary"]["phase_distribution"].items():
            txt_content += f"- {phase}: {duration:.2f}s\n"
        
        txt_content += "\nSentiment Distribution:\n"
        for sentiment, count in conversation.analysis["summary"]["sentiment_summary"].items():
            txt_content += f"- {sentiment}: {count}\n"
        
        txt_content += "\nTurn Taking Analysis:\n"
        turn_analysis = conversation.analysis["turn_taking"]
        txt_content += f"Total Turns: {turn_analysis['total_turns']}\n"
        txt_content += f"Salesperson Turns: {turn_analysis['salesperson_stats']['total_turns']}\n"
        txt_content += f"Customer Turns: {turn_analysis['customer_stats']['total_turns']}\n"
        
        return Response(
            content=txt_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=conversation_{conversation_id}.txt"
            }
        )

async def transcribe_and_analyze(
    file_path: str,
    db: Session,
    user_id: int,
    task_id: str
):
    """
    Background task to handle transcription and analysis
    """
    try:
        # Update progress to converting stage
        progress_tracker.update_progress(task_id, 1)
        
        # Transcribe audio
        progress_tracker.update_progress(task_id, 2)
        transcript = await transcribe_audio(file_path)
        
        # Identify speakers
        speakers = speaker_identifier.identify_speakers(transcript["segments"])
        transcript["speakers"] = speakers
        
        # Analyze turn-taking patterns
        turn_analysis = speaker_identifier.analyze_turn_taking(
            speakers,
            transcript["segments"]
        )
        
        # Classify dialogue
        progress_tracker.update_progress(task_id, 3)
        analysis = await classify_dialogue(transcript)
        analysis["turn_taking"] = turn_analysis
        
        # Store results
        conversation = ConversationCreate(
            user_id=user_id,
            file_path=file_path,
            transcript=transcript,
            analysis=analysis
        )
        
        db_conversation = Conversation(**conversation.dict())
        db.add(db_conversation)
        db.commit()
        
        # Mark task as completed
        progress_tracker.update_progress(task_id, 4, "completed")
        
    except Exception as e:
        progress_tracker.update_progress(task_id, -1, "error", str(e))
        raise e

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)