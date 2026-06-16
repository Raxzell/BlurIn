import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st


APP_NAME = "BlurIn"
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "mpeg4"]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]


st.set_page_config(
    page_title=APP_NAME,
    page_icon="◐",
    layout="centered",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #101114;
            --surface: #17191f;
            --surface-soft: #1d2027;
            --border: #2b3038;
            --border-soft: #363c46;
            --text: #f4f5f7;
            --muted: #a7adb7;
            --muted-2: #7f8792;
            --accent: #b8bec8;
            --accent-hover: #c8ced6;
            --button-text: #101114;
            --radius: 18px;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .main {
            background: var(--bg);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        h1 {
            color: var(--text);
            letter-spacing: -0.045em;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }

        .subtitle {
            color: var(--muted-2);
            font-size: 1rem;
            line-height: 1.65;
            margin-top: -0.25rem;
            margin-bottom: 1.25rem;
        }

        .info-card {
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.15rem 1.25rem;
            margin: 1.1rem 0 1.35rem 0;
        }

        .info-title {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
        }

        .small-text {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.6;
        }

        .upload-label {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 0.55rem;
        }

        div[data-testid="stFileUploader"] > label {
            display: none;
        }

        div[data-testid="stFileUploader"] {
            background: transparent;
        }

        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            min-height: 118px;
            padding: 0.75rem;
            transition: all 0.18s ease;
        }

        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
            background: var(--surface-soft);
            border-color: var(--border-soft);
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
            color: var(--muted) !important;
        }

        div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div {
            color: var(--text) !important;
            font-weight: 600;
        }

        div[data-testid="stFileUploaderDropzone"] small {
            color: var(--muted-2) !important;
        }

        div[data-testid="stFileUploaderDropzone"] svg {
            color: var(--muted) !important;
            fill: var(--muted) !important;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: #20242d !important;
            color: var(--text) !important;
            border: 1px solid var(--border-soft) !important;
            border-radius: 14px !important;
            font-weight: 650 !important;
            padding: 0.55rem 0.95rem !important;
            min-height: 44px !important;
        }

        div[data-testid="stFileUploaderDropzone"] button:hover {
            background: #262b35 !important;
            border-color: #4b5360 !important;
            color: var(--text) !important;
        }

        div[data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            margin-top: 1.1rem;
            margin-bottom: 1.15rem;
        }

        div[data-testid="stExpander"] details {
            background: transparent;
            border: none;
        }

        div[data-testid="stExpander"] summary {
            color: var(--text);
            font-weight: 650;
        }

        div[data-testid="stExpanderDetails"] {
            background: var(--surface);
            border-top: 1px solid var(--border);
        }

        .stButton > button {
            width: 100%;
            background: var(--accent);
            color: var(--button-text);
            border: 1px solid transparent;
            border-radius: var(--radius);
            padding: 0.85rem 1rem;
            font-weight: 750;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            background: var(--accent-hover);
            color: var(--button-text);
            border-color: transparent;
        }

        .stButton > button:focus:not(:active) {
            color: var(--button-text);
            border-color: transparent;
            box-shadow: none;
        }

        div[data-testid="stAlert"] {
            background: var(--surface-soft);
            border: 1px solid var(--border);
            border-radius: 14px;
            color: var(--text);
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] span,
        div[data-testid="stMarkdownContainer"] label {
            color: inherit;
        }

        .stSlider label {
            color: var(--text) !important;
        }

        div[data-testid="stVideo"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }

        .stDownloadButton > button {
            width: 100%;
            background: var(--surface-soft);
            color: var(--text);
            border: 1px solid var(--border-soft);
            border-radius: var(--radius);
            padding: 0.85rem 1rem;
            font-weight: 700;
        }

        .stDownloadButton > button:hover {
            background: #252a33;
            color: var(--text);
            border-color: #4b5360;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_face_detector():
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Haar Cascade cannot be loaded automatically from the installed OpenCV package."
        )

    return detector


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Reference image cannot be read. Use JPG, JPEG, or PNG.")

    return image


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.read())
        return temp_video.name


def convert_to_browser_compatible_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise ValueError("Video was processed, but conversion for browser preview failed.")

    return output_path


def crop_single_face(image_bgr, detector):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        raise ValueError(
            "No clear face was detected in the reference photo. Use one sharp front-facing face photo."
        )

    if len(faces) > 1:
        raise ValueError(
            "Reference photo must contain exactly one face. Crop or change the photo first."
        )

    x, y, w, h = faces[0]

    pad_x = int(w * 0.20)
    pad_y = int(h * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray.shape[1], x + w + pad_x)
    y2 = min(gray.shape[0], y + h + pad_y)

    cropped_face = gray[y1:y2, x1:x2]

    if cropped_face.size == 0:
        raise ValueError("The detected face crop is invalid. Use a clearer reference photo.")

    return cropped_face


def detect_faces_in_frame(gray_frame, detector):
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(35, 35),
    )

    if len(faces) == 0:
        return []

    boxes = []

    for x, y, w, h in faces:
        boxes.append((int(x), int(y), int(x + w), int(y + h)))

    boxes.sort(
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        reverse=True,
    )

    return boxes


def expand_box(box, frame_width, frame_height, padding):
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame_width, x2 + padding)
    y2 = min(frame_height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def smooth_box(previous_box, current_box, alpha=0.72):
    if previous_box is None:
        return current_box

    px1, py1, px2, py2 = previous_box
    cx1, cy1, cx2, cy2 = current_box

    x1 = int(alpha * px1 + (1 - alpha) * cx1)
    y1 = int(alpha * py1 + (1 - alpha) * cy1)
    x2 = int(alpha * px2 + (1 - alpha) * cx2)
    y2 = int(alpha * py2 + (1 - alpha) * cy2)

    return x1, y1, x2, y2


def make_odd(value):
    value = int(value)
    return value if value % 2 == 1 else value + 1


def blur_region(frame, box, blur_strength):
    x_min, y_min, x_max, y_max = box
    face_roi = frame[y_min:y_max, x_min:x_max]

    if face_roi.size == 0:
        return frame

    kernel = make_odd(blur_strength)
    blurred_face = cv2.GaussianBlur(face_roi, (kernel, kernel), 0)
    frame[y_min:y_max, x_min:x_max] = blurred_face

    return frame


def get_sift_match_score(reference_gray, candidate_gray, sift, flann, ratio_threshold):
    if candidate_gray.size == 0:
        return 0

    kp_ref, des_ref = sift.detectAndCompute(reference_gray, None)
    kp_candidate, des_candidate = sift.detectAndCompute(candidate_gray, None)

    if des_ref is None or des_candidate is None:
        return 0

    if len(des_ref) < 2 or len(des_candidate) < 2:
        return 0

    matches = flann.knnMatch(des_ref, des_candidate, k=2)
    good_matches = []

    for match_pair in matches:
        if len(match_pair) != 2:
            continue

        first_match, second_match = match_pair

        if first_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(first_match)

    return len(good_matches)


def choose_target_face(
    gray_frame,
    face_boxes,
    img_ref_gray,
    sift,
    flann,
    ratio_threshold,
):
    if len(face_boxes) == 0:
        return None

    if len(face_boxes) == 1:
        return face_boxes[0]

    best_box = None
    best_score = -1

    for box in face_boxes:
        x1, y1, x2, y2 = box
        face_roi = gray_frame[y1:y2, x1:x2]

        score = get_sift_match_score(
            reference_gray=img_ref_gray,
            candidate_gray=face_roi,
            sift=sift,
            flann=flann,
            ratio_threshold=ratio_threshold,
        )

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


def process_video(
    reference_image_bgr,
    video_path,
    output_path,
    progress_callback,
    status_callback,
    min_match_count=10,
    min_track_points=5,
    blur_strength=71,
    padding=45,
    detection_interval=12,
    ratio_threshold=0.7,
):
    detector = load_face_detector()
    img_ref_gray = crop_single_face(reference_image_bgr, detector)

    sift = cv2.SIFT_create()

    flann_index_kdtree = 1
    index_params = dict(
        algorithm=flann_index_kdtree,
        trees=5,
    )
    search_params = dict(
        checks=50,
    )
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video cannot be opened. Try MP4 with standard encoding.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError("Output video cannot be created on this system.")

    smoothed_box = None
    last_valid_box = None
    missing_face_frames = 0
    max_missing_face_frames = 5

    detected_frames = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_boxes = detect_faces_in_frame(curr_gray, detector)

        target_box = choose_target_face(
            gray_frame=curr_gray,
            face_boxes=face_boxes,
            img_ref_gray=img_ref_gray,
            sift=sift,
            flann=flann,
            ratio_threshold=ratio_threshold,
        )

        if target_box is not None:
            expanded_box = expand_box(
                box=target_box,
                frame_width=width,
                frame_height=height,
                padding=padding,
            )

            if expanded_box is not None:
                smoothed_box = smooth_box(
                    previous_box=smoothed_box,
                    current_box=expanded_box,
                    alpha=0.72,
                )

                last_valid_box = smoothed_box
                missing_face_frames = 0

                frame = blur_region(
                    frame=frame,
                    box=smoothed_box,
                    blur_strength=blur_strength,
                )

                detected_frames += 1

        else:
            missing_face_frames += 1

            if last_valid_box is not None and missing_face_frames <= max_missing_face_frames:
                frame = blur_region(
                    frame=frame,
                    box=last_valid_box,
                    blur_strength=blur_strength,
                )

                detected_frames += 1
            else:
                smoothed_box = None
                last_valid_box = None

        out.write(frame)
        processed_frames += 1

        if total_frames > 0:
            progress_callback(min(processed_frames / total_frames, 1.0))

        if processed_frames % 30 == 0:
            status_callback(
                f"Processed {processed_frames} frames. Blurred target in {detected_frames} frames."
            )

    cap.release()
    out.release()

    return {
        "processed_frames": processed_frames,
        "detected_frames": detected_frames,
    }


st.markdown(f"# {APP_NAME}")

st.markdown(
    """
    <p class="subtitle">
        Upload one reference face photo and one video. The matching person will be blurred in the output video.
    </p>
    """,
    unsafe_allow_html=True,
)


with st.container():
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Input rule</div>
            <div class="small-text">
                The reference photo must contain exactly one clear face. For better results, use a sharp front-facing image with good lighting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="upload-label">Reference face photo</div>',
        unsafe_allow_html=True,
    )

    reference_upload = st.file_uploader(
        "Reference face photo",
        type=SUPPORTED_IMAGE_TYPES,
        label_visibility="collapsed",
    )

with col2:
    st.markdown(
        '<div class="upload-label">Input video</div>',
        unsafe_allow_html=True,
    )

    video_upload = st.file_uploader(
        "Input video",
        type=SUPPORTED_VIDEO_TYPES,
        label_visibility="collapsed",
    )


with st.expander("Advanced settings"):
    min_match_count = st.slider(
        "Minimum SIFT matches",
        min_value=6,
        max_value=30,
        value=10,
    )

    min_track_points = st.slider(
        "Minimum tracking points",
        min_value=3,
        max_value=30,
        value=5,
    )

    blur_strength = st.slider(
        "Blur strength",
        min_value=21,
        max_value=101,
        value=71,
        step=2,
    )

    padding = st.slider(
        "Blur padding",
        min_value=10,
        max_value=90,
        value=45,
    )

    detection_interval = st.slider(
        "Re-detect every N frames",
        min_value=5,
        max_value=60,
        value=12,
    )

    ratio_threshold = st.slider(
        "Lowe ratio threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
    )


process_button = st.button("Process video", type="primary")


if process_button:
    if reference_upload is None or video_upload is None:
        st.error("Upload both the reference photo and the video first.")
    else:
        temp_video_path = None
        output_path = None
        browser_output_path = None

        try:
            reference_image = decode_uploaded_image(reference_upload)
            temp_video_path = save_uploaded_video(video_upload)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
                output_path = temp_output.name

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Processing video...")

            result = process_video(
                reference_image_bgr=reference_image,
                video_path=temp_video_path,
                output_path=output_path,
                progress_callback=progress_bar.progress,
                status_callback=status_text.info,
                min_match_count=min_match_count,
                min_track_points=min_track_points,
                blur_strength=blur_strength,
                padding=padding,
                detection_interval=detection_interval,
                ratio_threshold=ratio_threshold,
            )

            progress_bar.progress(1.0)

            status_text.success(
                f"Done. Processed {result['processed_frames']} frames and blurred the target in {result['detected_frames']} frames."
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as browser_output:
                browser_output_path = browser_output.name

            convert_to_browser_compatible_mp4(output_path, browser_output_path)

            with open(browser_output_path, "rb") as output_file:
                output_bytes = output_file.read()

            st.video(output_bytes)

            st.download_button(
                label="Download blurred video",
                data=output_bytes,
                file_name="blurin_output.mp4",
                mime="video/mp4",
            )

        except Exception as error:
            st.error(str(error))

        finally:
            for path in [temp_video_path, output_path, browser_output_path]:
                if path and os.path.exists(path):
                    os.remove(path)
