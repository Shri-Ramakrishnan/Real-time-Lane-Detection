

# === Real-Time Road Lane Detection for Autonomous Vehicles (Enhanced) ===
# Dataset: ashikadnan/driving-video-for-lane-detection-various-weather (KaggleHub)
# Author: ChatGPT GPT-5
# Dependencies: OpenCV, NumPy (both available in Colab)

import cv2
import numpy as np
import os
import time
import random
from IPython.display import HTML, display
from base64 import b64encode

# ============================================================
# CONFIGURATION
# ============================================================
base_dir = "/root/.cache/kagglehub/datasets/ashikadnan/driving-video-for-lane-detection-various-weather/versions/1/drivingDataset"

categories = {
    "rainyDay": ['rD_6.mp4', 'rD_16.mp4', 'rD_19.mp4', 'rD_1.mp4', 'rD_17.mp4', 'rD_15.mp4', 'rD_11.mp4', 'rD_14.mp4', 'rD_13.mp4', 'rD_8.mp4'],
    "normalDay": ['nD_10.mp4', 'nD_9.mp4', 'nD_17.mp4', 'nD_18.mp4', 'nD_7.mp4', 'nD_3.mp4', 'nD_5.mp4', 'ND_19.mp4', 'nD_16.mp4', 'nD_2.mp4'],
    "normalNight": ['nN_10.mp4', 'nN_16.mp4', 'nN_15.mp4', 'nN_8.mp4', 'nN_13.mp4', 'nN_7.mp4', 'nN_18.mp4', 'nN_12.mp4', 'nN_5.mp4', 'nN_14.mp4']
}

# Choose one category
category = "normalDay"  # Change to "rainyDay" or "normalNight"
video_name = random.choice(categories[category])

input_video = os.path.join(base_dir, category, video_name)
output_video = f"lane_detected_{category}_improved.mp4"

print(f"🎥 Using video: {input_video}")
print(f"💾 Output will be saved as: {output_video}")

# ============================================================
# PARAMETERS / TUNING
# ============================================================
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW, CANNY_HIGH = 50, 150
SMOOTHING_FRAMES = 5

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def save_video_check(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges

def region_of_interest(img):
    h, w = img.shape[:2]
    hist = np.sum(img[h//2:h, :], axis=1)
    horizon = np.argmax(hist) + h//2
    horizon = min(horizon, int(0.75*h))
    bottom_left = (int(0.1*w), h)
    bottom_right = (int(0.9*w), h)
    top_left = (int(0.35*w), horizon)
    top_right = (int(0.65*w), horizon)
    polygons = np.array([[bottom_left, bottom_right, top_right, top_left]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def get_perspective_transform_matrices(frame):
    h, w = frame.shape[:2]
    src = np.float32([
        [int(0.45*w), int(0.62*h)],
        [int(0.55*w), int(0.62*h)],
        [int(0.9*w), h],
        [int(0.1*w), h]
    ])
    dst = np.float32([
        [int(0.2*w), 0],
        [int(0.8*w), 0],
        [int(0.8*w), h],
        [int(0.2*w), h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp(img, M, size):
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    margin, minpix = 100, 50
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    lefty = nonzeroy[left_lane_inds] if left_lane_inds.size > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if right_lane_inds.size > 0 else np.array([])
    righty = nonzeroy[right_lane_inds] if right_lane_inds.size > 0 else np.array([])

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2) if leftx.size else None
    right_fit = np.polyfit(righty, rightx, 2) if rightx.size else None
    return left_fit, right_fit

def smooth_fit(history, new_fit, alpha=0.2):
    if new_fit is None:
        return np.mean(history, axis=0) if history else None
    if not history:
        history.append(new_fit)
    else:
        history[-1] = alpha * new_fit + (1 - alpha) * history[-1]
    return history[-1]

# --- Enhanced lane visualization (full green fill) ---
def draw_lane(original_img, binary_warped, Minv, left_fit, right_fit):
    h, w = original_img.shape[:2]
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    color_warp = np.zeros_like(original_img).astype(np.uint8)

    if left_fit is None or right_fit is None:
        return original_img

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create a polygon to cover the entire area between lanes
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the lane area fully with bright green
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.4, 0)
    return result

def calculate_lane_deviation(left_fit, right_fit, frame_shape):
    h, w = frame_shape[:2]
    y_eval = h
    if left_fit is None or right_fit is None:
        return 0
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    vehicle_center = w / 2
    return vehicle_center - lane_center

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_video(input_path, output_path):
    save_video_check(output_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_path}")

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

    M, Minv, size = None, None, (frame_width, frame_height)
    left_history, right_history = [], []
    start_time, frame_count = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        edges = preprocess_frame(frame)
        masked = region_of_interest(edges)

        if M is None:
            M, Minv = get_perspective_transform_matrices(frame)
        warped = warp(masked, M, size)
        binary_warped = (warped > 0).astype(np.uint8) * 255

        left_fit, right_fit = fit_polynomial(binary_warped)
        left_fit_avg = smooth_fit(left_history, left_fit)
        right_fit_avg = smooth_fit(right_history, right_fit)

        lane_image = draw_lane(frame, binary_warped, Minv, left_fit_avg, right_fit_avg)
        deviation = calculate_lane_deviation(left_fit_avg, right_fit_avg, frame.shape)
        direction = "Left" if deviation < -10 else "Right" if deviation > 10 else "Center"

        cv2.putText(lane_image, f"Frame: {frame_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(lane_image, f"Offset: {deviation:.1f}px ({direction})", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        out.write(lane_image)

    cap.release()
    out.release()
    duration = time.time() - start_time
    avg_fps = frame_count / duration if duration > 0 else 0
    print(f"✅ Processed {frame_count} frames in {duration:.1f}s — approx {avg_fps:.2f} FPS")
    print("💾 Saved to:", os.path.abspath(output_path))
    return avg_fps

# ============================================================
# RUN PIPELINE
# ============================================================
fps_result = process_video(input_video, output_video)


# Re-encode to H.264 + AAC and move moov atom to start (faststart)
import subprocess, shlex, os, time

src = "/content/lane_detected_normalDay_improved.mp4"   # <--- change if your filename differs
fixed = "/content/lane_detected_normalDay_fixed.mp4"

# remove existing fixed file (avoid prompts)
if os.path.exists(fixed):
    os.remove(fixed)

cmd = f'ffmpeg -y -i "{src}" -c:v libx264 -preset fast -crf 23 -movflags +faststart -c:a aac -b:a 128k "{fixed}"'

print("Running ffmpeg re-encode (this may take some minutes)...")
start = time.time()
proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
end = time.time()

print("ffmpeg exit code:", proc.returncode)
print("Elapsed:", round(end - start,1), "s")
print("\nLast 800 chars of ffmpeg stderr (useful if something went wrong):\n")
print(proc.stderr.decode(errors='ignore')[-800:])

print("\nFixed file exists:", os.path.exists(fixed))
if os.path.exists(fixed):
    print("Fixed size (MB):", round(os.path.getsize(fixed)/(1024*1024), 2))
else:
    print("Fixed file not created.")



from IPython.display import Video, display, HTML
fixed = "/content/lane_detected_normalDay_fixed.mp4"

if os.path.exists(fixed):
    print("Playing fixed file:", fixed)
    display(Video(fixed, embed=True, width=720, height=480))
    display(HTML(f'<p><a href="/content/{os.path.basename(fixed)}" download>⬇️ Download processed video (fixed)</a></p>'))
else:
    print("Fixed file not found. Run the re-encode step first.")

