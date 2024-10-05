
#!/bin/bash

# Set the project name
PROJECT_NAME="lightglue-fastapi-app"

# Set the required python version
PYTHON_VERSION="3.9"

# Create the project directory if it doesn't exist
mkdir -p $PROJECT_NAME

# Change into the project directory
cd $PROJECT_NAME

# Create the app directory
mkdir -p app
cd app

# Create the main.py file
touch main.py

# Create the utils.py file
touch utils.py

# Create the templates directory and add the html templates
mkdir -p templates
cd templates
touch index.html
touch upload.html
touch webcam.html
cd ..

# Create the static directory and add the css and js files
mkdir -p static
cd static
touch styles.css
touch webcam.js
cd ..

# Create the assets directory and add a placeholder for image assets
cd ..
mkdir -p assets

# Create a placeholder for the LightGlue repository
mkdir -p LightGlue

# Create the requirements.txt file
touch requirements.txt

# Create the README.md file
touch README.md

# Change back to the project root
cd ..

# Create a virtual environment with the required python version
python3 -m venv --python=${PYTHON_VERSION} venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt