# BlurIn

BlurIn is a Streamlit-based web application that automatically blurs a selected person in a video using one reference face photo.

The app takes two inputs:

1. A reference face photo containing exactly one clear face
2. A video containing one or more people

BlurIn then processes the video and applies blur to the person whose face matches the uploaded reference photo.

## Fun Fact About the Name

**BlurIn** comes from the phrase **"blur in"** or **"di-blur-in"**, meaning the selected person is blurred inside the video.

There is also a small wordplay: because the uppercase **I** and lowercase **l** can look similar, **BlurIn** can visually resemble **"Biarin"** in Indonesian, which means **"let it be"** or **"leave it alone"**. So the name carries both a technical meaning and a casual Indonesian twist.

## Features

* Upload one reference face photo
* Upload one input video
* Automatically detect and crop the face from the reference photo
* Match the reference face inside the video
* Blur the matched person in the output video
* Download the processed video
* Adjustable advanced settings for matching, tracking, blur strength, and detection interval
* Clean semi-monochrome Streamlit interface

## Tech Stack

* Python
* Streamlit
* OpenCV
* NumPy

## How It Works

BlurIn uses a computer vision pipeline built with OpenCV:

1. **Face Detection on Reference Image**
   The uploaded reference photo is processed using OpenCV Haar Cascade. The app requires the reference image to contain exactly one clear face.

2. **Feature Extraction with SIFT**
   The cropped reference face is converted into visual keypoints and descriptors using SIFT.

3. **Feature Matching with FLANN**
   Each video frame is checked against the reference face using FLANN-based matching and Lowe's ratio test.

4. **Homography Estimation**
   When enough matching features are found, homography is used to estimate the target face region in the video frame.

5. **Tracking with Lucas-Kanade Optical Flow**
   After the target is detected, the app tracks the region using Lucas-Kanade optical flow to reduce the need for full detection on every frame.

6. **Gaussian Blur**
   The detected or tracked region is blurred using Gaussian blur and written into the output video.

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
```

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
6. Preview and download the blurred output video.

## Recommended Input

For better results, use:

* A sharp reference photo
* A front-facing face image
* Good lighting
* A video where the target face is not too small
* A video with minimal extreme face rotation or heavy occlusion

## Advanced Settings

The app provides several adjustable parameters:

### Minimum SIFT Matches

Controls how many visual feature matches are required before the app considers the target detected.

Higher value means stricter matching, but detection may be slower or less sensitive.

### Minimum Tracking Points

Controls how many optical flow points are required to keep tracking the target.

Higher value can reduce unstable tracking, but may lose the target more easily.

### Blur Strength

Controls the intensity of the Gaussian blur applied to the target region.

### Blur Padding

Expands the blur area around the detected/tracked region.

Useful for making sure the whole face is covered.

### Re-detect Every N Frames

Controls how often the app performs SIFT-based re-detection while tracking.

Lower value means more frequent re-detection, which can reduce tracking drift but may increase processing time.

### Lowe Ratio Threshold

Controls the strictness of feature matching.

Lower value means stricter matching and fewer false matches.

## Suggested Settings

For general use:

```text
Minimum SIFT matches: 10
Minimum tracking points: 5
Blur strength: 51
Blur padding: 35
Re-detect every N frames: 8–12
Lowe ratio threshold: 0.70
```

If the blur follows the wrong background object:

```text
Increase Minimum tracking points to 8
Decrease Re-detect every N frames to 5–8
Increase Minimum SIFT matches to 12–16
```

If the target face is detected too slowly:

```text
Decrease Minimum SIFT matches to 8–10
Set Re-detect every N frames to 5–8
Keep Lowe ratio threshold around 0.70
```

## Limitations

BlurIn uses classical computer vision methods, not a deep learning face recognition model.

This means the app is based on visual feature matching and tracking rather than modern face embeddings. Because of that, performance may decrease when:

* The face is too small
* Lighting changes significantly
* The face is heavily rotated
* The face is covered
* The person leaves the frame
* The background has similar visual features
* The video is shaky or blurry

In some cases, the blur may briefly follow background objects after the target leaves the frame because optical flow tracks visual points, not identity.

## Notes

This project is intended as a computer vision application demo using OpenCV-based techniques such as Haar Cascade, SIFT, FLANN, Homography, Lucas-Kanade Optical Flow, and Gaussian Blur.

## License

This project is open for educational and personal use.
