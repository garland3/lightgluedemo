# app/main.py

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from utils import match_images, image_to_bytes
import io
import cv2
import numpy as np
import base64  # Add this import at the top

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Assets directory
ASSETS_DIR = Path("static/assets")
# make if not exists
if not ASSETS_DIR.exists():
    ASSETS_DIR.mkdir()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Home page with options to upload images or use the webcam.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """
    Page to upload two images for matching.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_images(request: Request, file1: UploadFile = File(None), file2: UploadFile = File(None), use_defaults: str = Form(None)):
    """
    Handle uploaded images or use defaults, perform matching, and display results.
    """
    if use_defaults:
        image_path1 = ASSETS_DIR / "1.png"
        image_path2 = ASSETS_DIR / "2.png"
        # print a message 
        print("Using default images")
    else:
        print("Using uploaded images")
        if file1 is None or file2 is None:
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "matched": False,
                "error": "Please upload both images or select the option to use defaults."
            })
        print(file1.filename, file2.filename)
        # Save uploaded images
        contents1 = await file1.read()
        contents2 = await file2.read()
        image_path1 = ASSETS_DIR / file1.filename
        image_path2 = ASSETS_DIR / file2.filename
        print("Image paths:", image_path1, image_path2)

        # Save uploaded images
        with open(image_path1, "wb") as f:
            f.write(contents1)
        with open(image_path2, "wb") as f:
            f.write(contents2)

    # Perform matching
    results = match_images(image_path1, image_path2)

    # Convert images to bytes
    image1_bytes = image_to_bytes(results["image0"])
    image2_bytes = image_to_bytes(results["image1"])

    # Encode images to base64
    image1_base64 = base64.b64encode(image1_bytes).decode('utf-8')
    image2_base64 = base64.b64encode(image2_bytes).decode('utf-8')

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "matched": True,
        "stop_layers": results["matches"]["stop_layers"],
        "image1": f"data:image/jpeg;base64,{image1_base64}",
        "image2": f"data:image/jpeg;base64,{image2_base64}",
    })


@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request):
    """
    Page to use the webcam for real-time feature matching.
    """
    return templates.TemplateResponse("webcam.html", {"request": request})

def generate_frames():
    """
    Generator function to capture frames from the webcam, perform matching, and yield them as a video stream.
    """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame if necessary
        # For simplicity, we're not performing matching here
        # Implement real-time matching as needed

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """
    Video streaming route. Put this in the src attribute of an img tag.
    """
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
