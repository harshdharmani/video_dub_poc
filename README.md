# AI Video Dubbing POC

## 🎯 Project Objective
Build a Proof of Concept (POC) system that takes a video in one language (English) and outputs the same video dubbed into another language (Hindi) using AI.

## 🧠 High-Level Pipeline
1. **Video (.mp4)** → **Audio Extraction**
2. **Audio** → **Speech-to-Text (ASR)**
3. **Text** → **Text Translation**
4. **Translated Text** → **Text-to-Speech**
5. **Speech** → **Merge Audio with Video**
6. **Output** → **Dubbed Video**

## 🧩 Technologies & Models

### 1. Audio / Video Processing
- **FFmpeg**: Used for extracting audio from video and merging generated audio back. (System dependency)

### 2. Speech-to-Text (ASR)
- **Deepgram Nova-3**: Cloud-based ASR for high accuracy and speed.
- **Input**: `audio/original.wav`
- **Output**: English transcript
- **Requirement**: `DEEPGRAM_API_KEY` environment variable.

### 3. Translation (Planned)
- **Helsinki-NLP / MarianMT**: `opus-mt-en-hi`
- Transformer-based Neural Machine Translation.
- Runs offline.

### 4. Text-to-Speech (Planned)
- **gTTS (Google Text-to-Speech)**: Simple and reliable for POC.
- **Output**: Hindi speech (.mp3)

## 📁 Project Structure
```
Voice_dubb_poc/
│
├── input/
│   └── sample.mp4
├── audio/
│   └── original.wav
├── output/
│   └── dubbed.mp4
├── core/
│   ├── audioextracter.py
│   ├── transcribe.py
│   ├── translator.py      (In Progress)
│   └── tts.py             (In Progress)
├── main.py
├── requirements.txt
└── .env
```

## ⚙️ Setup & Usage

### 1. Environment Variables
Ensure you have a Deepgram API key set:
```bash
export DEEPGRAM_API_KEY=your_api_key_here
# OR (Windows PowerShell)
setx DEEPGRAM_API_KEY "your_api_key_here"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: FFmpeg must be installed on your system path.*

### 3. Run the Pipeline
```bash
python main.py
```

## 🚀 Status
- [x] FFmpeg Pipeline Setup
- [x] Audio Extraction (`core/audioextracter.py`)
- [x] Transcription with Deepgram (`core/transcribe.py`)
- [ ] Translation Logic
- [ ] Text-to-Speech Logic
- [ ] Video Merging
