import gradio as gr
from matplotlib import pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import cv2
import numpy as np
import warnings

# Suppress FutureWarnings from lightglue and torch (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create LightGlue matcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)


# def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
#     image = read_image(path)
#     if resize is not None:
#         image, _ = resize_image(image, resize, **kwargs)
#     return numpy_image_to_torch(image)



def match_images(img1, img2):
    # Load images
    
    # img1 and img2 are ndarrays, convert to cv2
    # image0 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    # image1 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    image0  = numpy_image_to_torch(img1)
    image1 = numpy_image_to_torch(img2)
    
    # image0 = load_image(img1)
    # image1 = load_image(img2)

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
    version = 1
    if version ==1:
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    if version ==2:
        # Plot pruned keypoints
        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

    # Create a single image with the matches and pruned keypoints
    # image = viz2d.savefig()
    # save the plt figure. 
    plt.savefig('temp.png')
    # load the image as PIL. 
    return 'temp.png'
    # image = cv2.imread('temp.png')
    # image = cv2.imdecode(np.array(bytearray(image), dtype=np.uint8), cv2.IMREAD_COLOR)

    # return image


# Create Gradio app
demo = gr.Blocks()
with demo:
    gr.Markdown("# LightGlue Demo")
    
    gr.Markdown("## Match two images")
    with gr.Row():
        img1 = gr.Image(label="Image 1")
        img2 = gr.Image(label="Image 2")
    match_but = gr.Button("Match")
    img_example = gr.Image(label="Matches and pruned keypoints")

    match_but.click(match_images, inputs=[img1, img2], outputs=img_example)

    

demo.launch()
