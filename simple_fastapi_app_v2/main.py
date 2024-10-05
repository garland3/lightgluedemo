from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import base64
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch
from lightglue import viz2d
import io
from matplotlib import pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize LightGlue matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# Global variable to store the reference image
reference_image = None

def match_images(img1, img2):
    # Convert images to torch tensors
    image0 = numpy_image_to_torch(img1)
    image1 = numpy_image_to_torch(img2)

    # Extract features
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # Match images
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # Remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Get keypoints and matches
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Plot matches
    # axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
  # Plot matches
    version = 2
    if version ==1:
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    if version ==2:
        # Plot pruned keypoints
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set_reference")
async def set_reference(request: Request):
    global reference_image
    form = await request.form()
    image = form["image"].file.read()
    nparr = np.frombuffer(image, np.uint8)
    reference_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logger.info("Reference image set")
    return {"message": "Reference image set"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection opened")
    try:
        while True:
            data = await websocket.receive_text()
            logger.info("Received frame from client")
            if reference_image is None:
                await websocket.send_text("Reference image not set")
                logger.warning("Reference image not set")
                continue
            
            # Decode base64 image
            try:
                img_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                await websocket.send_text("Error decoding image")
                continue

            # Match images
            try:
                result_buf = match_images(reference_image, img)
            except Exception as e:
                logger.error(f"Error matching images: {e}")
                await websocket.send_text("Error matching images")
                continue

            # Encode result image to base64
            result_base64 = base64.b64encode(result_buf.getvalue()).decode('utf-8')
            await websocket.send_text(f"data:image/png;base64,{result_base64}")
            logger.info("Sent matched image to client")
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)