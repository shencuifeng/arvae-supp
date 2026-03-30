#!/usr/bin/env python3
"""Merge videos side-by-side with model-name labels burned into each sub-clip."""
import subprocess
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "iclr_video", "merged")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

TARGET_H = 256
LABEL_H = 28  # height of the label bar on top of each sub-video


def get_font(size=16):
    """Try to load a nice font, fall back to default."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ]
    for fp in font_candidates:
        if os.path.isfile(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def add_label_bar_to_video(input_path, output_path, label, target_h=TARGET_H):
    """
    Read a video, scale to target_h, add a white label bar on top with text,
    and write to output_path using cv2.
    """
    # cv2.VideoCapture cannot handle non-ASCII paths; use a temp copy if needed
    ascii_tmp = None
    open_path = input_path
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        # Try copying to an ASCII-safe temp path via ffmpeg
        ascii_tmp = os.path.join(TMP_DIR, f"_ascii_{os.getpid()}_{id(input_path)}.mp4")
        copy_cmd = [FFMPEG, "-y", "-i", input_path, "-c", "copy", ascii_tmp]
        copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=120)
        if copy_result.returncode != 0:
            print(f"  ERROR: Cannot open or copy {input_path}")
            print(f"         ffmpeg stderr: {copy_result.stderr[-300:]}")
            return False, 0
        open_path = ascii_tmp
        cap = cv2.VideoCapture(open_path)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open even after copy: {open_path}")
            return False, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    font = get_font(15)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Scale to target height
        original_h, original_w = frame.shape[:2]
        scale = target_h / original_h
        new_w = int(original_w * scale)
        # Force even width
        new_w = new_w if new_w % 2 == 0 else new_w + 1
        frame = cv2.resize(frame, (new_w, target_h))

        # Create label bar (white background)
        bar = np.ones((LABEL_H, new_w, 3), dtype=np.uint8) * 255

        # Draw text using PIL for better rendering
        bar_pil = Image.fromarray(bar)
        draw = ImageDraw.Draw(bar_pil)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (new_w - text_w) // 2
        text_y = (LABEL_H - text_h) // 2 - bbox[1]
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        bar = np.array(bar_pil)

        # Stack bar on top of frame
        combined = np.vstack([bar, frame])
        frames.append(combined)

    cap.release()

    # Clean up ASCII temp copy if created
    if ascii_tmp and os.path.isfile(ascii_tmp):
        try:
            os.remove(ascii_tmp)
        except OSError:
            pass

    if not frames:
        print(f"  ERROR: No frames read from {input_path}")
        return False, 0

    total_h, total_w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (total_w, total_h))
    for f in frames:
        writer.write(f)
    writer.release()
    return True, total_w


def merge_videos(inputs, labels, output_name, target_h=TARGET_H):
    """
    Merge multiple videos horizontally with text labels burned in.
    """
    output_path = os.path.join(OUT_DIR, output_name)

    full_paths = []
    for inp in inputs:
        fp = os.path.join(BASE, inp)
        if not os.path.isfile(fp):
            print(f"  WARNING: Missing {fp}, skipping this group")
            return False
        full_paths.append(fp)

    num_videos = len(full_paths)

    # Step 1: Add label bar to each video individually
    labeled_paths = []
    for i, (fp, label) in enumerate(zip(full_paths, labels)):
        tmp_path = os.path.join(TMP_DIR, f"{output_name}_part{i}.mp4")
        ok, width = add_label_bar_to_video(fp, tmp_path, label, target_h)
        if not ok:
            return False
        labeled_paths.append(tmp_path)

    # Step 2: Use ffmpeg to hstack the labeled videos
    filter_parts = []
    stack_inputs = []
    for i in range(num_videos):
        filter_parts.append(f"[{i}:v]setsar=1,format=yuv420p[s{i}]")
        stack_inputs.append(f"[s{i}]")

    stack_str = "".join(stack_inputs)
    filter_parts.append(f"{stack_str}hstack=inputs={num_videos}[out]")
    filter_complex = ";".join(filter_parts)

    cmd = [FFMPEG, "-y"]
    for lp in labeled_paths:
        cmd.extend(["-i", lp])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        "-shortest",
        output_path,
    ])

    print(f"  Merging -> {output_name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False

    # Clean up temp files
    for lp in labeled_paths:
        try:
            os.remove(lp)
        except OSError:
            pass

    print(f"  OK: {output_path}")
    return True


# ============================================================
# Define all video groups (excluding zero-out and latent swap)
# ============================================================

print("=== 8x Reconstruction ===")
for clip_num in range(1, 5):
    clip = f"clip{clip_num}"
    d = f"iclr_video/8x video results/{clip}"

    if clip_num == 1:
        vidtok = f"{d}/vidtok-kl.mp4"
        vidtok_label = "VidTok-KL"
    else:
        vidtok = f"{d}/vidtok_{clip_num}.mp4"
        vidtok_label = "VidTok"

    if clip_num == 1:
        cogvideox = f"{d}/cogvideox.mp4"
        videovaeplus = f"{d}/videovaeplus.mp4"
    else:
        cogvideox = f"{d}/cogvideox_{clip_num}.mp4"
        videovaeplus = f"{d}/videvaeplus_{clip_num}.mp4" if clip_num == 2 else f"{d}/videovaeplus_{clip_num}.mp4"

    merge_videos(
        inputs=[
            f"{d}/groundtruth.mp4",
            f"{d}/ARVAE(ours)_{clip_num}.mp4",
            cogvideox,
            videovaeplus,
            vidtok,
        ],
        labels=["Ground Truth", "ARVAE (Ours)", "CogVideoX", "VideoVAEPlus", vidtok_label],
        output_name=f"8x_{clip}.mp4",
    )

print("\n=== 16x Reconstruction ===")
for clip_num in range(1, 6):
    clip = f"clip{clip_num}"
    d = f"iclr_video/16x video results/{clip}"

    if clip_num == 4:
        step_file = f"{d}/step_4.mp4"
    elif clip_num == 5:
        step_file = f"{d}/step.mp4"
    else:
        step_file = f"{d}/stepvideo.mp4"

    gt_file = f"{d}/gt.mp4" if clip_num == 5 else f"{d}/groundtruth.mp4"

    merge_videos(
        inputs=[gt_file, f"{d}/ARVAE(ours).mp4", step_file],
        labels=["Ground Truth", "ARVAE (Ours)", "StepVideo"],
        output_name=f"16x_{clip}.mp4",
    )

print("\n=== 32x Reconstruction ===")
for clip_num in range(1, 5):
    clip = f"clip{clip_num}"
    d = f"iclr_video/32x video results/{clip}"

    ltx_file = f"{d}/ltx_4.mp4" if clip_num == 4 else f"{d}/ltx.mp4"

    merge_videos(
        inputs=[f"{d}/groundtruth.mp4", f"{d}/ARVAE(ours).mp4", ltx_file],
        labels=["Ground Truth", "ARVAE (Ours)", "LTX"],
        output_name=f"32x_{clip}.mp4",
    )

print("\n=== Generation: Sky ===")
merge_videos(
    inputs=[
        "iclr_video/generation video results/sky/ours.mp4",
        "iclr_video/generation video results/sky/ours1.mp4",
        "iclr_video/generation video results/sky/ours2.mp4",
        "iclr_video/generation video results/sky/digan(Comparison method\uff09.mp4",
    ],
    labels=["ARVAE (Ours)", "ARVAE (Ours) #2", "ARVAE (Ours) #3", "DIGAN"],
    output_name="gen_sky.mp4",
)

print("\n=== Generation: Taichi ===")
merge_videos(
    inputs=[
        "iclr_video/generation video results/Taichi/ours.mp4",
        "iclr_video/generation video results/Taichi/ours1.mp4",
        "iclr_video/generation video results/Taichi/digan\uff08Comparison method\uff09.mp4",
        "iclr_video/generation video results/Taichi/diagn1\uff08Comparison method\uff09.mp4",
    ],
    labels=["ARVAE (Ours)", "ARVAE (Ours) #2", "DIGAN", "DIGAN #2"],
    output_name="gen_taichi.mp4",
)

# Clean up tmp dir
import shutil
try:
    shutil.rmtree(TMP_DIR)
except OSError:
    pass

print("\n=== DONE ===")
print(f"Merged videos saved to: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".mp4"):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}: {sz / 1024:.0f} KB")