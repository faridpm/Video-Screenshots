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

st.caption("Tip: For large videos (over 500 MB), split into 2–3 parts to keep each download manageable.")

with st.expander("Crop settings (recommended: remove browser bar, macOS bar & webcam panel)"):
    st.markdown("Cut away the macOS menu bar, browser bar, and webcam panel on the right so only the shared screen content is kept.")
    use_crop = st.checkbox("Enable crop")
    if use_crop:
        crop_col1, crop_col2 = st.columns(2)
        with crop_col1:
            crop_top = st.slider("Cut from top (%)", 0, 40, 19)
            crop_left = st.slider("Cut from left (%)", 0, 40, 0)
        with crop_col2:
            crop_bottom = st.slider("Cut from bottom (%)", 0, 40, 8)
            crop_right = st.slider("Cut from right (%)", 0, 40, 13)
    else:
        crop_top = crop_bottom = crop_left = crop_right = 0

with st.expander("Contact Sheet option (recommended for Mural uploads)"):
    st.markdown("**What is a Contact Sheet?**\nInstead of hundreds of individual images, screenshots are combined into wide rows (e.g. 5 per row). This keeps them in the correct left-to-right order when uploading to tools like **Mural**.")
    use_contact_sheet = st.checkbox("Create Contact Sheets instead of individual screenshots")
    if use_contact_sheet:
        per_row = st.select_slider("Screenshots per row", options=[3, 4, 5, 6], value=5)
    else:
        per_row = 5

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])


def crop_frame(frame, top, bottom, left, right):
    h, w = frame.shape[:2]
    return frame[int(h*top/100):int(h*(100-bottom)/100), int(w*left/100):int(w*(100-right)/100)]


def resize_frame(frame, max_width=800):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    return frame


if uploaded_file:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"**{uploaded_file.name}** — {file_size_mb:.1f} MB")

    if st.button("Create screenshots", type="primary"):
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

            st.write(f"Duration: **{duration_s/60:.1f} min** | FPS: **{fps:.1f}** | Expected: **~{int(duration_s/interval)} screenshots**")

            interval_frames = max(1, int(interval * fps))
            frames_per_part = ceil(total_frames / split_parts)
            progress_bar = st.progress(0.0)
            status = st.empty()

            parts = {p: [] for p in range(1, split_parts + 1)}
            screenshot_count = 0
            skipped_count = 0
            frame_number = 0
            last_saved_gray = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % interval_frames == 0:
                    processed = crop_frame(frame, crop_top, crop_bottom, crop_left, crop_right) if use_crop else frame
                    processed = resize_frame(processed, max_width=800)
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

                    is_duplicate = last_saved_gray is not None and np.mean(cv2.absdiff(last_saved_gray, gray)) < 1.5

                    if not is_duplicate:
                        part_num = min(frame_number // frames_per_part + 1, split_parts)
                        ts = frame_number / fps
                        ts_str = f"{int(ts//3600):02d}h{int((ts%3600)//60):02d}m{int(ts%60):02d}s"
                        _, img_bytes = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        parts[part_num].append((f"screenshot_{screenshot_count:04d}_{ts_str}.jpg", img_bytes.tobytes()))
                        screenshot_count += 1
                        last_saved_gray = gray
                    else:
                        skipped_count += 1

                frame_number += 1
                if frame_number % 300 == 0:
                    progress_bar.progress(min(frame_number / total_frames, 1.0))
                    status.text(f"{frame_number/total_frames*100:.0f}% processed — {screenshot_count} screenshots, {skipped_count} duplicates skipped")

            cap.release()
            progress_bar.progress(1.0)
            status.text(f"Done! {screenshot_count} screenshots created, {skipped_count} duplicates skipped.")
            st.success(f"{screenshot_count} screenshots are ready!")

            for part_num in range(1, split_parts + 1):
                part_screenshots = parts[part_num]
                if not part_screenshots:
                    continue

                if use_contact_sheet:
                    sheets_buffer = io.BytesIO()
                    sheet_count = 0
                    with zipfile.ZipFile(sheets_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for i in range(0, len(part_screenshots), per_row):
                            row_data = part_screenshots[i:i+per_row]
                            row_frames = []
                            for _, img_bytes in row_data:
                                arr = np.frombuffer(img_bytes, np.uint8)
                                row_frames.append(cv2.imdecode(arr, cv2.IMREAD_COLOR))
                            while len(row_frames) < per_row:
                                row_frames.append(np.zeros_like(row_frames[0]))
                            sheet = np.hstack(row_frames)
                            _, sheet_bytes = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            zf.writestr(f"contact_sheet_{sheet_count+1:03d}.jpg", sheet_bytes.tobytes())
                            sheet_count += 1
                    sheets_buffer.seek(0)
                    part_label = f" (part {part_num} of {split_parts})" if split_parts > 1 else ""
                    st.download_button(
                        f"Download {sheet_count} contact sheets{part_label} as ZIP",
                        sheets_buffer,
                        file_name=f"contact_sheets_part{part_num}.zip" if split_parts > 1 else "contact_sheets.zip",
                        mime="application/zip",
                        key=f"sheet_{part_num}",
                    )
                else:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fname, data in part_screenshots:
                            zf.writestr(fname, data)
                    zip_buffer.seek(0)
                    label = f"Download part {part_num} of {split_parts} ({len(part_screenshots)} screenshots)" if split_parts > 1 else f"Download {screenshot_count} screenshots as ZIP"
                    st.download_button(
                        label, zip_buffer,
                        file_name=f"screenshots_part{part_num}.zip" if split_parts > 1 else "screenshots.zip",
                        mime="application/zip",
                        key=f"part_{part_num}",
                    )

        finally:
            os.unlink(tmp_path)
