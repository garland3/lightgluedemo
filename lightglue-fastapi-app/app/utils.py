# app/utils.py

from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
import numpy as np
import cv2
from typing import Tuple, Dict

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize extractor and matcher
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

def match_images(image_path1: Path, image_path2: Path) -> Dict:
    """
    Match two images using LightGlue and SuperPoint.

    Args:
        image_path1 (Path): Path to the first image.
        image_path2 (Path): Path to the second image.

    Returns:
        Dict: A dictionary containing matched keypoints and other metadata.
    """
    image0 = load_image(image_path1)
    image1 = load_image(image_path2)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return {
        "image0": image0.cpu().numpy(),
        "image1": image1.cpu().numpy(),
        "matches": {
            "keypoints0": kpts0.cpu().numpy(),
            "keypoints1": kpts1.cpu().numpy(),
            "matches": matches.cpu().numpy(),
            "matched_keypoints0": m_kpts0.cpu().numpy(),
            "matched_keypoints1": m_kpts1.cpu().numpy(),
            "stop_layers": matches01["stop"],
            "prune0": matches01["prune0"].cpu().numpy(),
            "prune1": matches01["prune1"].cpu().numpy(),
        }
    }

def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert a NumPy image array to bytes for sending over HTTP.

    Args:
        image (np.ndarray): Image array.

    Returns:
        bytes: JPEG-encoded image.
    """
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()
