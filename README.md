# Sales Conversation Analysis System

A system for transcribing and analyzing sales conversations using locally hosted Whisper model.

## Features

- Audio file upload and processing (WAV, MP3, M4A)
- Local speech-to-text using Whisper
- Speaker identification
- Conversation phase classification
- Sentiment analysis
- Progress tracking
- Search functionality

## Requirements

### Hardware
- Minimum 16GB RAM
- GPU support recommended
- 500GB storage
- Stable internet connection

### Software
See `requirements.txt` for Python dependencies.

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run the application:
```bash
python run.py
```

## API Documentation

Once running, access the API documentation at:
http://localhost:8000/docs

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── main.py                 # FastAPI application
│   ├── models/                 # Database models
│   │   └── database.py
│   ├── schemas/               # Pydantic schemas
│   │   └── conversation.py
│   └── services/             # Business logic
│       ├── audio_processing.py
│       ├── auth.py
│       ├── classification.py
│       ├── progress.py
│       ├── search.py
│       ├── speaker_identification.py
│       └── transcription.py
└── uploads/                  # Audio file storage
```

## Security

- Local data storage only
- User authentication required
- JWT-based access control
- Data encryption in transit
- Audit logging

## License

MIT