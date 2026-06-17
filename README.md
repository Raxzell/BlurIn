# BlurIn

BlurIn is a Streamlit-based web application that automatically blurs a selected person in a video using one reference face photo.

The app takes two inputs:

1. A reference face photo containing exactly one clear face
2. A video containing one or more people

BlurIn processes the video, finds the person whose face matches the uploaded reference photo, tracks the target across frames, and applies blur to the matched face in the output video.

## Fun Fact About the Name

**BlurIn** comes from the phrase **"blur in"** or **"di-blur-in"**, meaning the selected person is blurred inside the video.

There is also a small wordplay: because the uppercase **I** and lowercase **l** can look similar, **BlurIn** can visually resemble **"Biarin"** in Indonesian, which means **"let it be"** or **"leave it alone"**. So the name carries both a technical meaning and a casual Indonesian twist.

## Features

* Upload one reference face photo
* Upload one input video
* Automatically detect and crop the face from the reference photo
* Match the reference face inside the video
* Track the matched face across video frames
* Blur the matched person in the output video
* Preview the processed video in the browser
* Download the processed video as an MP4 file
* Adjustable advanced settings for matching, tracking, blur strength, padding, and re-detection interval
* Clean semi-monochrome Streamlit interface

## Tech Stack

* Python
* Streamlit
* OpenCV
* NumPy
* imageio-ffmpeg

## How It Works

BlurIn uses a computer vision pipeline built with OpenCV:

1. **Face Detection on Reference Image**  
   The uploaded reference photo is processed using OpenCV Haar Cascade. The app requires the reference image to contain exactly one clear face.

2. **Reference Face Cropping**  
   The detected face region is cropped with additional padding so the system can extract more useful facial features.

3. **Face Detection on Video Frames**  
   Each video frame is converted to grayscale, then Haar Cascade is used to detect candidate faces inside the frame.

4. **Feature Extraction with SIFT**  
   The cropped reference face and candidate faces from the video are converted into visual keypoints and descriptors using SIFT.

5. **Feature Matching with FLANN**  
   The descriptors from the reference face and candidate faces are matched using a FLANN-based matcher.

6. **Lowe's Ratio Test**  
   Lowe's Ratio Test is used to filter weak or unreliable matches so the system only keeps stronger visual matches.

7. **Homography Estimation**  
   When enough matching features are found, Homography Estimation with RANSAC is used to validate and estimate the target face region. If the homography result is unstable, the app falls back to the detected face bounding box.

8. **Lucas-Kanade Optical Flow Tracking**  
   After the target is detected, Lucas-Kanade Optical Flow is used to track visual points inside the target face region across frames. This reduces the need for full face re-detection and feature matching on every frame.

9. **Periodic Re-detection**  
   The app periodically re-detects the target face based on the selected re-detection interval. This helps reduce tracking drift and keeps the blur aligned with the target face.

10. **Bounding Box Smoothing**  
   The blur region is smoothed across frames to reduce shaking or sudden jumps in the blur area.

11. **Gaussian Blur**  
   The final face region is blurred using Gaussian Blur and written into the output video.

12. **MP4 Browser Conversion**  
   The processed video is converted into a browser-compatible MP4 format using imageio-ffmpeg and FFmpeg settings such as H.264 encoding and yuv420p pixel format.

## Project Structure

```text
BlurIn/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/BlurIn.git
cd BlurIn
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```txt
streamlit==1.41.1
opencv-contrib-python-headless==4.10.0.84
numpy==1.26.4
imageio-ffmpeg
```

> Important: use `opencv-contrib-python-headless`, not only `opencv-python`, because this project uses `cv2.SIFT_create()`.

## Run the App

```bash
streamlit run app.py
```

After running the command, open the local Streamlit URL shown in the terminal.

## Usage

1. Upload a reference face photo.
2. Make sure the reference photo contains exactly one clear face.
3. Upload an input video.
4. Adjust advanced settings if needed.
5. Click **Process video**.
6. Preview the processed video.
7. Download the blurred output video.

## Recommended Input

For better results, use:

* A sharp reference photo
* A front-facing face image
* Good lighting
* A video where the target face is not too small
* A video with minimal extreme face rotation
* A video with minimal heavy occlusion
* A stable video with limited motion blur

## Advanced Settings

The app provides several adjustable parameters:

### Minimum SIFT Matches

Controls how many visual feature matches are required before the app considers the target detected.

Higher value means stricter matching and fewer false matches, but detection may become less sensitive.

Recommended value:

```text
10–16
```

### Minimum Tracking Points

Controls how many optical flow points are required to keep tracking the target.

Higher value can reduce unstable tracking, but may lose the target more easily when the face moves quickly or becomes unclear.

Recommended value:

```text
8–12
```

### Blur Strength

Controls the intensity of the Gaussian blur applied to the target region.

Higher value means stronger blur.

Recommended value:

```text
71
```

### Blur Padding

Expands the blur area around the detected or tracked face region.

Useful for making sure the whole face is covered.

Recommended value:

```text
25–35
```

If the blur area becomes too large or unstable, reduce this value.

### Re-detect Every N Frames

Controls how often the app performs full target re-detection using Haar Cascade, SIFT, FLANN, and Homography validation.

Lower value means more frequent re-detection, which can reduce tracking drift but may increase processing time.

Recommended value:

```text
5–12
```

### Lowe Ratio Threshold

Controls the strictness of SIFT feature matching.

Lower value means stricter matching and fewer false matches. Higher value means more permissive matching but may increase incorrect matches.

Recommended value:

```text
0.65–0.70
```

## Suggested Settings

For general use:

```text
Minimum SIFT matches: 12
Minimum tracking points: 8
Blur strength: 71
Blur padding: 30
Re-detect every N frames: 8
Lowe ratio threshold: 0.70
```

If the blur follows the wrong person or background object:

```text
Increase Minimum SIFT matches to 14–18
Decrease Lowe ratio threshold to 0.60–0.65
Decrease Re-detect every N frames to 5–8
Increase Minimum tracking points to 10–12
```

If the blur area becomes too large:

```text
Decrease Blur padding to 20–30
Increase Minimum tracking points to 8–12
Decrease Re-detect every N frames to 5–8
```

If the target face is detected too slowly:

```text
Decrease Minimum SIFT matches to 8–10
Set Re-detect every N frames to 5–8
Keep Lowe ratio threshold around 0.70
```

## Output

After processing, BlurIn provides:

* A preview of the processed video
* A downloadable MP4 file named `blurin_output.mp4`
* Processing status information, including:
  * Total processed frames
  * Number of blurred frames
  * Number of re-detections
  * Number of frames tracked using Lucas-Kanade Optical Flow

## Limitations

BlurIn uses classical computer vision methods, not a deep learning face recognition model.

This means the app is based on visual feature matching, face detection, homography validation, and optical flow tracking rather than modern face embeddings. Because of that, performance may decrease when:

* The face is too small
* Lighting changes significantly
* The face is heavily rotated
* The face is covered
* The person leaves the frame
* The background has similar visual features
* The video is shaky or blurry
* The target face has very few clear visual features
* Multiple people have similar facial appearance or similar lighting conditions

In some cases, the blur may drift or briefly follow the wrong region because Lucas-Kanade Optical Flow tracks visual points, not identity. The app uses periodic re-detection to reduce this issue, but it cannot fully guarantee perfect tracking in every video.

## Notes

This project is intended as a computer vision application demo using OpenCV-based techniques such as Haar Cascade, SIFT, FLANN, Homography Estimation, Lucas-Kanade Optical Flow, and Gaussian Blur.

For more accurate real-world face anonymization, a deep learning-based face detector and face recognition model may produce better results.

## License

This project is open for educational and personal use.
