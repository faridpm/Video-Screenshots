import streamlit as st
import cv2
import numpy as np
import os
import zipfile
import tempfile
import io
from pathlib import Path
from math import ceil

st.set_page_config(page_title="Video → Screenshots", layout="centered")

st.title("Video → Screenshots")
st.markdown("Upload a video and automatically receive screenshots as a ZIP download.")

# --- Settings ---
col1, col2 = st.columns(2)

with col1:
    interval = st.selectbox(
        "Interval between screenshots",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: f"every {x} second{'s' if x > 1 else ''}",
    )

with col2:
    split_parts = st.selectbox(
        "Split video into",
        options=[1, 2, 3, 4],
        index=0,
        format_func=lambda x: "no splitting" if x == 1 else f"{x} parts",
    )

st.caption(
    "Tip: For large videos (over 500 MB), split into 2–3 parts "
    "to keep each download manageable."
)

# --- Upload ---
uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Supported formats: MP4, MOV, AVI, MKV",
)

if uploaded_file:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"**{uploaded_file.name}** — {file_size_mb:.1f} MB")

    if st.button("Create screenshots", type="primary"):

        # Save uploaded file to temp file
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("Could not open video. Please try a different format.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_s = total_frames / fps

            st.write(
                f"Duration: **{duration_s/60:.1f} min** &nbsp;|&nbsp; "
                f"FPS: **{fps:.1f}** &nbsp;|&nbsp; "
                f"Expected: **~{int(duration_s / interval)} screenshots**"
            )

            interval_frames = max(1, int(interval * fps))
            frames_per_part = ceil(total_frames / split_parts)

            progress_bar = st.progress(0.0)
            status = st.empty()

            # Collect screenshots in memory, split by part
            parts: dict[int, list[tuple[str, bytes]]] = {p: [] for p in range(1, split_parts + 1)}
            screenshot_count = 0
            skipped_count = 0
            frame_number = 0
            last_saved_gray: np.ndarray | None = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % interval_frames == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Duplicate check
                    if last_saved_gray is not None:
                        diff = np.mean(cv2.absdiff(last_saved_gray, gray))
                        is_duplicate = diff < 1.5
                    else:
                        is_duplicate = False

                    if not is_duplicate:
                        part_num = min(frame_number // frames_per_part + 1, split_parts)
                        ts = frame_number / fps
                        ts_str = f"{int(ts//3600):02d}h{int((ts%3600)//60):02d}m{int(ts%60):02d}s"
                        fname = f"screenshot_{screenshot_count:04d}_{ts_str}.png"

                        _, img_bytes = cv2.imencode(".png", frame)
                        parts[part_num].append((fname, img_bytes.tobytes()))
                        screenshot_count += 1
                        last_saved_gray = gray
                    else:
                        skipped_count += 1

                frame_number += 1

                if frame_number % 300 == 0:
                    progress = frame_number / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    status.text(
                        f"{progress*100:.0f}% processed — "
                        f"{screenshot_count} screenshots, {skipped_count} duplicates skipped"
                    )

            cap.release()
            progress_bar.progress(1.0)
            status.text(
                f"Done! {screenshot_count} screenshots created, {skipped_count} duplicates skipped."
            )

            # Create ZIP(s) and show download buttons
            st.success(f"{screenshot_count} screenshots are ready!")

            for part_num in range(1, split_parts + 1):
                part_screenshots = parts[part_num]
                if not part_screenshots:
                    continue

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, data in part_screenshots:
                        zf.writestr(fname, data)
                zip_buffer.seek(0)

                label = (
                    f"Download part {part_num} of {split_parts} ({len(part_screenshots)} screenshots)"
                    if split_parts > 1
                    else f"Download {screenshot_count} screenshots as ZIP"
                )
                zip_name = f"screenshots_part{part_num}.zip" if split_parts > 1 else "screenshots.zip"

                st.download_button(
                    label=label,
                    data=zip_buffer,
                    file_name=zip_name,
                    mime="application/zip",
                    key=f"download_part_{part_num}",
                )

        finally:
            os.unlink(tmp_path)
