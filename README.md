Computer Vision Techniques API

Overview

This project is a FastAPI-based web application that demonstrates various computer vision techniques, including feature detection, feature matching, and face detection. The API provides endpoints for processing images using different algorithms and returning annotated images.

Features

Corner Detection:

Harris Corner Detector

Shi-Tomasi Corner Detector

FAST Detector

ORB Detector

Feature Matching:

Brute Force Matching

FLANN Matching

RANSAC for outlier detection

Face Detection:

Uses Haar Cascades for face detection

Installation

Prerequisites

Ensure you have the following installed:

Python 3.8+

pip (or pip3)

uvicorn (for running the FastAPI server)

Setup

Clone the repository:

git clone https://github.com/your-username/Computer-Vision-Techniques-API.git
cd Computer-Vision-Techniques-API

Install dependencies:

pip install -r requirements.txt

Running the API

Start the FastAPI server using Uvicorn:

uvicorn main:app --reload

The API will be accessible at: http://127.0.0.1:8000

API Endpoints

1. Feature Detection

Harris Corner Detection

Endpoint: /detect_features/harris

Method: POST

Input: Image file

Output: Processed image with detected corners

Shi-Tomasi Corner Detection

Endpoint: /detect_features/shi_tomasi

FAST Corner Detection

Endpoint: /detect_features/fast

ORB Feature Detection

Endpoint: /detect_features/orb

2. Feature Matching

Brute Force Matching

Endpoint: /feature_matching/brute_force

Input: Two image files

Output: Image showing matched keypoints

FLANN Matching

Endpoint: /feature_matching/flann

3. Face Detection

Haar Cascade Face Detection

Endpoint: /face_detection

Input: Image file

Output: Image with detected faces

Usage Example

Using cURL to send an image to the API:

curl -X 'POST' \
  'http://127.0.0.1:8000/detect_features/harris' \
  -F 'file=@image.jpg'

Contributing

Feel free to fork the repository and submit pull requests with improvements or bug fixes.

License

This project does not currently have a license. You may use or modify it as needed.
