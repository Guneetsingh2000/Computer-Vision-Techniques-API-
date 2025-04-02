from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
from typing import Optional
from io import BytesIO
from fastapi.responses import JSONResponse, Response

app = FastAPI()

def read_image(file: UploadFile):
    """Reads an uploaded image file and converts it to OpenCV format."""
    image_bytes = file.file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

@app.post("/detect_features/harris")
def harris_corner_detector(file: UploadFile = File(...), block_size: int = 2, ksize: int = 3, k: float = 0.04):
    image = read_image(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red
    _, encoded_img = cv2.imencode(".jpg", image)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/detect_features/shi_tomasi")
def shi_tomasi_detector(file: UploadFile = File(...), max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10):
    image = read_image(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    if corners is not None:
        for i in np.int0(corners):
            x, y = i.ravel()
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Mark corners in green
    _, encoded_img = cv2.imencode(".jpg", image)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/detect_features/fast")
def fast_detector(file: UploadFile = File(...), threshold: int = 10):
    image = read_image(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold)
    keypoints = fast.detect(gray, None)
    result_img = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))
    _, encoded_img = cv2.imencode(".jpg", result_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/detect_features/orb")
def orb_detector(file: UploadFile = File(...), nfeatures: int = 500):
    image = read_image(file)
    orb = cv2.ORB_create(nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    result_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    _, encoded_img = cv2.imencode(".jpg", result_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/feature_matching/brute_force")
def brute_force_matching(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image(file1)
    img2 = read_image(file2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _, encoded_img = cv2.imencode(".jpg", matched_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/feature_matching/flann")
def flann_matching(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image(file1)
    img2 = read_image(file2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # FLANN parameters for matching
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # or pass in a smaller number
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _, encoded_img = cv2.imencode(".jpg", matched_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/feature_matching/ransac")
def ransac_matching(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image(file1)
    img2 = read_image(file2)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Convert matches to points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    
    # Only keep good matches (inliers)
    matches_mask = mask.ravel().tolist()
    good_matches = [m for m, mask in zip(matches, matches_mask) if mask]
    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    _, encoded_img = cv2.imencode(".jpg", matched_img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.post("/face_detection")
def face_detection(file: UploadFile = File(...), scale_factor: float = 1.1, min_neighbors: int = 5):
    image = read_image(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use OpenCV’s built-in Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    _, encoded_img = cv2.imencode(".jpg", image)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

