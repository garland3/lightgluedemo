# LightGlue FastAPI Application

This application demonstrates feature matching between images using LightGlue and SuperPoint, wrapped in a FastAPI web application. Users can either upload two images for feature matching or use their webcam for real-time frame tracking.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Upload Images](#upload-images)
  - [Use Webcam](#use-webcam)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- **Image Upload**: Upload two images and visualize feature matches using LightGlue.
- **Webcam Streaming**: Stream webcam video (extension for real-time matching can be implemented).
- **Interactive Web Interface**: User-friendly interface built with HTML and CSS.

## Prerequisites

- **Python 3.8+**
- **Git**
- **CUDA** (Optional, for GPU acceleration with PyTorch)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/lightglue-fastapi-app.git
   cd lightglue-fastapi-app




```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```