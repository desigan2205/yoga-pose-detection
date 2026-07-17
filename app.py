import os

import cv2
import numpy as np
from PIL import Image

import streamlit as st

try:
    import av
    AV_AVAILABLE = True
except Exception:
    AV_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

import config
from detection import PoseDetector
from feedback import build_full_feedback

st.set_page_config(
    page_title="Yoga Pose Detection",
    page_icon="",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        from keypoint_inference import get_keypoint_predictor
        predictor = get_keypoint_predictor()
        return predictor, None
    except Exception:
        pass
    try:
        from model_predict import get_predictor
        predictor = get_predictor()
        return predictor, None
    except Exception as exc:
        return None, str(exc)


def render_sidebar():
    conf_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0,
        value=config.DEFAULT_CONF_THRESHOLD,
        step=0.05,
    )
    show_skeleton = st.sidebar.checkbox("Show skeleton overlay", value=True)
    model_quality = st.sidebar.selectbox(
        "Pose model quality:",
        ["lite (fast)", "full (balanced)", "heavy (accurate)"],
        index=1,
        help="Full model recommended for best balance of speed & accuracy",
    )
    return conf_threshold, show_skeleton, model_quality


def render_top3(top3):
    st.markdown("#### Top 3 Predictions")
    for name, prob in top3:
        pct = prob * 100
        st.write(f"**{name}** — {pct:.2f}%")
        bar = "🟩" if pct > 50 else "🟨" if pct > 20 else "🟥"
        st.write(bar * int(pct / 10) + "⬜" * (10 - int(pct / 10)))


def main():
    st.title(" Yoga Pose Detection")
    st.caption("Upload an image or use your webcam to detect yoga poses in real time.")

    conf_threshold, show_skeleton, model_quality = render_sidebar()
    model_quality = model_quality.split(" ")[0]  # "lite", "full", or "heavy"

    predictor, err = load_predictor()
    if err:
        st.warning(
            " Model not available. Skeleton detection still works, "
            "but pose classification is disabled.\n\n"
            f"Reason: {err}"
        )

    mode = st.radio("Select input mode:", ["Upload Image", "Live Webcam"], horizontal=True)

    if mode == "Upload Image":
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
                rgb = np.array(pil_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                st.error("Could not read image.")
                st.stop()

            display_img = bgr.copy()
            landmarks = None
            found = False
            if show_skeleton:
                try:
                    with PoseDetector(static_image_mode=True, model_quality=model_quality) as det:
                        display_img, found = det.find_pose(display_img, draw=True)
                        if found:
                            landmarks = det.get_landmarks(bgr)
                except Exception as exc:
                    st.warning(f"Skeleton detection failed: {exc}")

            st.image(
                cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                caption="Skeleton overlay" if (show_skeleton and found) else "Input image",
                use_container_width=True,
            )

            if not found:
                st.info("No human pose detected in this image.")
                st.stop()

            if predictor is None:
                st.info("Pose detected! Train a model to get pose classification.")
                st.stop()

            try:
                if hasattr(predictor, "landmarks_to_features"):
                    name, conf, top3 = predictor.predict(bgr, landmarks)
                else:
                    name, conf, top3 = predictor.predict(bgr)
                fb = build_full_feedback(name, conf)

                if conf >= conf_threshold:
                    st.success(f"**{name}**")
                else:
                    st.info(f"Best guess: **{name}** (below {conf_threshold*100:.0f}% threshold)")

                st.progress(min(max(conf, 0.0), 1.0))
                st.write(f"**Confidence:** {conf * 100:.2f}%")
                st.write(f"{fb['emoji']} {fb['message']} — *{fb['tip']}*")
                render_top3(top3)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:
        if not WEBRTC_AVAILABLE:
            st.error("Live camera requires `streamlit-webrtc`.\nInstall: `pip install streamlit-webrtc av`")
            st.stop()

        st.info("Press **START** to begin. Press **STOP** to end.")

        cam_option = st.selectbox("Camera preference:", ["Front (default)", "Back (rear)"], index=0)
        facing_mode = "user" if cam_option == "Front (default)" else "environment"

        res_option = st.selectbox(
            "Video quality:",
            ["High (720p)", "Standard (480p)", "Low (360p)"],
            index=0,
        )
        res_map = {
            "High (720p)": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
            "Standard (480p)": {"width": {"ideal": 854}, "height": {"ideal": 480}},
            "Low (360p)": {"width": {"ideal": 640}, "height": {"ideal": 360}},
        }
        video_constraints = {**res_map[res_option], "facingMode": facing_mode}

        class YogaVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.detector = PoseDetector(static_image_mode=False, model_quality=model_quality)
                self.predictor = predictor
                self.conf_threshold = conf_threshold
                self.show_skeleton = show_skeleton

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                landmarks = None
                try:
                    if self.show_skeleton:
                        img, pose_found = self.detector.find_pose(img, draw=True)
                        if pose_found:
                            landmarks = self.detector.get_landmarks(img)
                        else:
                            cv2.putText(img, "No person detected", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Skeleton overlay OFF", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if self.predictor is not None:
                        if hasattr(self.predictor, "landmarks_to_features"):
                            name, conf, _ = self.predictor.predict(img, landmarks)
                        else:
                            name, conf, _ = self.predictor.predict(img)
                        fb = build_full_feedback(name, conf)
                        label = f"{name} ({conf*100:.0f}%)" if conf >= self.conf_threshold else "Uncertain..."
                        overlay = img.copy()
                        cv2.rectangle(overlay, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.85, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, fb["message"], (12, img.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="yoga-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YogaVideoProcessor,
            media_stream_constraints={"video": video_constraints, "audio": False},
            async_processing=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  ️ Poses")
    for i, p in enumerate(config.POSES, 1):
        st.sidebar.write(f"{i}. {p}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
