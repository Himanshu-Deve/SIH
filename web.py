import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

class YOLOTransform(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        return results[0].plot()

st.title("Plant Disease Web Detection")
webrtc_streamer(
    key="camera",
    video_transformer_factory=YOLOTransform,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
