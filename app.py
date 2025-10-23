import streamlit as st
import tempfile
import os
import io
import matplotlib.pyplot as plt
from gtts import gTTS
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm # Use standard tqdm for non-notebook environments

# Set Streamlit Page Config
st.set_page_config(page_title="Pose Comparison App", layout="wide")

# Initialize MediaPipe Pose
@st.cache_resource
def get_pose_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose

pose = get_pose_model()
mp_pose = mp.solutions.pose

# Define body parts (moved to top-level for better scope)
body_parts = {
    "left_arm": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
    "right_arm": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
    "left_leg": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
    "right_leg": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    "torso": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
              mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]
}

# 🔹 Utility: Detect if a file is an image (by Streamlit's mime-type)
def is_image_file(file_object):
    return file_object.type.startswith('image/')

# 🔹 Extract keypoints (supports both videos and images)
def extract_keypoints(path, is_video, frame_skip=5, max_frames=300):
    keypoints = []
    
    if not is_video:
        # Handle image input
        frame = cv2.imread(path)
        if frame is None:
            st.warning("Could not read image file.")
            return np.array(keypoints)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z if lm.HasField('z') else 0) for lm in results.pose_landmarks.landmark]
            keypoints.append(landmarks)
    else:
        # Handle video input
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use st.progress for Streamlit environment
        progress_bar = st.progress(0, text="Extracting keypoints...")
        
        frame_count = 0
        processed_frames = 0
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    # Include Z-coordinate if available
                    landmarks = [(lm.x, lm.y, lm.z if lm.HasField('z') else 0) for lm in results.pose_landmarks.landmark]
                    keypoints.append(landmarks)
                processed_frames += frame_skip
                progress = min(1.0, processed_frames / max_frames)
                progress_bar.progress(progress, text=f"Processing frame {processed_frames}/{total_frames}...")
            frame_count += 1
        cap.release()
        progress_bar.empty() # Clear the progress bar
        
    return np.array(keypoints)

# 🔹 Compare two sequences (DTW) - No changes needed, logic remains the same
def compare_sequences(ref_kp, user_kp, body_parts):
    # Ensure keypoints are 3D (frames, landmarks, coords)
    if ref_kp.ndim == 2:
        ref_kp = np.expand_dims(ref_kp, axis=0)
    if user_kp.ndim == 2:
        user_kp = np.expand_dims(user_kp, axis=0)

    # Use coordinates (x, y, z) for DTW
    coord_dim = ref_kp.shape[-1]
    
    part_scores = {}
    frame_distances = {}
    
    min_frames = min(ref_kp.shape[0], user_kp.shape[0])

    for part, indices in body_parts.items():
        # Reshape to (min_frames, num_landmarks * coord_dim)
        ref_part_seq = ref_kp[:min_frames, indices, :coord_dim].reshape(min_frames, -1)
        user_part_seq = user_kp[:min_frames, indices, :coord_dim].reshape(min_frames, -1)
        
        # FastDTW
        distance, path = fastdtw(ref_part_seq, user_part_seq, dist=euclidean)
        
        # Simple scoring (may need calibration)
        score = max(0, 100 - distance * 10)
        part_scores[part] = round(score, 2)

        # Per-frame distances
        per_frame_dist = []
        for i in range(min_frames):
            frame_dist = euclidean(ref_part_seq[i], user_part_seq[i])
            per_frame_dist.append(frame_dist)
        frame_distances[part] = per_frame_dist
        
    return part_scores, frame_distances

# 🔹 Visualization
def draw_pose_on_frame(frame, kp_user, kp_ref=None, color_bgr=(0, 255, 0)):
    overlay = frame.copy()
    height, width, _ = frame.shape
    pose_connections = list(mp_pose.POSE_CONNECTIONS)
    
    for connection in pose_connections:
        p1_idx, p2_idx = connection
        
        # Check for user keypoints
        if p1_idx < len(kp_user) and p2_idx < len(kp_user):
            ux1, uy1, _ = kp_user[p1_idx]
            ux2, uy2, _ = kp_user[p2_idx]
            
            # Draw user lines
            cv2.line(overlay, (int(ux1 * width), int(uy1 * height)),
                     (int(ux2 * width), int(uy2 * height)), color_bgr, 2)
            
            # Draw reference keypoints as small circles for comparison
            if kp_ref is not None:
                 if p1_idx < len(kp_ref) and p2_idx < len(kp_ref):
                    rx1, ry1, _ = kp_ref[p1_idx]
                    rx2, ry2, _ = kp_ref[p2_idx]
                    cv2.circle(overlay, (int(rx1 * width), int(ry1 * height)), 5, (0, 0, 255), -1) # Red dots for reference
                    cv2.circle(overlay, (int(rx2 * width), int(ry2 * height)), 5, (0, 0, 255), -1)

    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


# ------------------- Streamlit UI -------------------
st.title("🤸‍♂ AI Pose Comparison and Feedback")
st.markdown("Upload a *Reference* image/video and your *User* image/video to get an accuracy score using Dynamic Time Warping (DTW) on MediaPipe keypoints.")

col1, col2 = st.columns(2)

with col1:
    reference_file = st.file_uploader("Upload Reference File (Image/Video)", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])
with col2:
    user_file = st.file_uploader("Upload User File (Image/Video)", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

st.markdown("---")

if st.button("Start Comparison", type="primary"):
    if reference_file is None or user_file is None:
        st.error("Please upload both a Reference file and a User file.")
    else:
        # Use tempfile to handle the uploaded file bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_file.name)[1]) as ref_temp:
            ref_temp.write(reference_file.read())
            ref_path = ref_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_file.name)[1]) as user_temp:
            user_temp.write(user_file.read())
            user_path = user_temp.name
            
        ref_is_video = not is_image_file(reference_file)
        user_is_video = not is_image_file(user_file)

        try:
            # 1. Keypoint Extraction
            with st.spinner("Extracting keypoints from files..."):
                ref_kp = extract_keypoints(ref_path, ref_is_video)
                user_kp = extract_keypoints(user_path, user_is_video)

            if ref_kp.size == 0 or user_kp.size == 0:
                st.error("❌ Could not detect a human pose in one or both files. Try another input.")
            else:
                st.success("✅ Keypoint extraction complete.")

                # 2. Sequence Comparison
                with st.spinner("Comparing pose sequences (DTW)..."):
                    scores, frame_distances = compare_sequences(ref_kp, user_kp, body_parts)
                st.success("✅ Comparison complete!")
                
                # 3. Display Results
                st.subheader("Comparison Results")
                results_cols = st.columns(len(scores))
                
                overall_score = sum(scores.values()) / len(scores)
                
                # Display individual scores
                for idx, (part, score) in enumerate(scores.items()):
                    results_cols[idx].metric(label=f"Avg. {part.replace('_', ' ').title()} Score", value=f"{score:.2f}%")
                    
                st.markdown(f"## ⭐ Overall Accuracy: *{overall_score:.2f}%*")
                st.progress(overall_score / 100)

                # 4. Generate Verbal Feedback
                feedback_text = f"Your performance analysis is as follows: Overall accuracy is {overall_score:.2f} percent. "
                for part, score in scores.items():
                    feedback_text += f"The {part.replace('_', ' ')} scored {score} percent. "
                
                tts = gTTS(text=feedback_text, lang='en')
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                st.audio(mp3_fp, format='audio/mp3', autoplay=True)
                st.info("Listen to the verbal feedback above.")
                
                # 5. Visualization (Only for single image comparison)
                if not ref_is_video and not user_is_video:
                    st.subheader("Image Pose Comparison Visualization")
                    frame_ref = cv2.imread(ref_path)
                    frame_user = cv2.imread(user_path)
                    
                    # Ensure frames are available and not None
                    if frame_ref is not None and frame_user is not None:
                        # Draw user pose (green) and reference keypoints (red dots) on user image
                        annotated_user = draw_pose_on_frame(frame_user, user_kp[0], ref_kp[0], color_bgr=(0, 255, 0))
                        
                        # Convert BGR to RGB for matplotlib/streamlit
                        annotated_rgb = cv2.cvtColor(annotated_user, cv2.COLOR_BGR2RGB)
                        
                        st.image(annotated_rgb, caption="User Pose with Reference Keypoints (Red Dots)", use_column_width=True)
                    else:
                        st.warning("Could not load image files for visualization.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            # Clean up temporary files
            os.remove(ref_path)
            os.remove(user_path)
            st.info("Temporary files cleaned up.")