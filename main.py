import os
import time
import shutil
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core.pipeline import process_video
from core.translator import SUPPORTED_LANGUAGES

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

# Templates
templates = Jinja2Templates(directory="templates")

# Ensure directories exist
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": SUPPORTED_LANGUAGES
    })

@app.post("/process", response_class=HTMLResponse)
async def process_dubbing(
    request: Request,
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    video_file: UploadFile = File(None),
    youtube_url: str = Form(None)
):
    upload_time = 0.0
    download_time = 0.0
    video_path = ""
    
    # Validation
    if not video_file and not youtube_url:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please upload a video or provide a YouTube link.",
            "languages": SUPPORTED_LANGUAGES
        })

    try:
        if youtube_url:
            # Handle YouTube URL
            import yt_dlp
            print(f"Downloading YouTube URL: {youtube_url}")
            t0 = time.time()
            
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': 'input/%(title)s.%(ext)s',
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                filename = ydl.prepare_filename(info)
                video_path = filename
            
            download_time = time.time() - t0
            print(f"Download finished: {video_path} in {download_time:.2f}s")
            
        elif video_file:
            # Handle File Upload
            t0 = time.time()
            video_path = f"input/{video_file.filename}"
            print(f"Saving uploaded file to: {video_path}")
            
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video_file.file, buffer)
            
            upload_time = time.time() - t0
            print(f"Upload finished in {upload_time:.2f}s")

        # Process Video
        result = process_video(video_path, source_lang, target_lang)
        
        # Prepare context for result page
        return templates.TemplateResponse("result.html", {
            "request": request,
            "upload_time": upload_time,
            "download_time": download_time,
            "timings": result["timings"],
            "transcription": result["transcription"],
            "output_video": f"/output/{os.path.basename(result['output_video_path'])}",
            "source_lang": source_lang,
            "target_lang": target_lang
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing video: {str(e)}",
            "languages": SUPPORTED_LANGUAGES
        })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
