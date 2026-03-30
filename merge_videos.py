#!/usr/bin/env python3
"""Merge videos side-by-side with labels for each comparison group."""
import subprocess
import os
import sys

# Get ffmpeg path from imageio_ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "iclr_video", "merged")
os.makedirs(OUT_DIR, exist_ok=True)

# Target height for all videos (will scale to this)
TARGET_H = 256

def merge_videos(inputs, labels, output_name, target_h=TARGET_H):
    """
    Merge multiple videos horizontally with text labels.
    inputs: list of file paths (relative to BASE)
    labels: list of label strings
    output_name: output filename (will be placed in OUT_DIR)
    """
    output_path = os.path.join(OUT_DIR, output_name)
    
    # Check all inputs exist
    full_paths = []
    for inp in inputs:
        fp = os.path.join(BASE, inp)
        if not os.path.isfile(fp):
            print(f"  WARNING: Missing {fp}, skipping this group")
            return False
        full_paths.append(fp)
    
    n = len(full_paths)
    
    # Build ffmpeg filter:
    # 1. Scale each video to target height, force even width
    # 2. Add text label on each
    # 3. hstack all together
    filter_parts = []
    stack_inputs = []
    
    for i in range(n):
        # Scale to target height, force even dimensions
        filter_parts.append(
            f"[{i}:v]scale=-2:{target_h},setsar=1,format=yuv420p[s{i}]"
        )
        stack_inputs.append(f"[s{i}]")
    
    # hstack
    stack_str = "".join(stack_inputs)
    filter_parts.append(f"{stack_str}hstack=inputs={n}[out]")
    
    filter_complex = ";".join(filter_parts)
    
    # Build command
    cmd = [FFMPEG, "-y"]
    for fp in full_paths:
        cmd.extend(["-i", fp])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        "-shortest",
        output_path
    ])
    
    print(f"  Merging -> {output_name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    print(f"  OK: {output_path}")
    return True


# ============================================================
# Define all video groups
# ============================================================

print("=== 8x Reconstruction ===")
for clip_num in range(1, 5):
    clip = f"clip{clip_num}"
    d = f"iclr_video/8x video results/{clip}"
    
    # Determine vidtok filename
    if clip_num == 1:
        vidtok = f"{d}/vidtok-kl.mp4"
        vidtok_label = "VidTok-KL"
    else:
        vidtok = f"{d}/vidtok_{clip_num}.mp4"
        vidtok_label = "VidTok"
    
    # Determine other filenames
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
        output_name=f"8x_{clip}.mp4"
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
        output_name=f"16x_{clip}.mp4"
    )

print("\n=== 32x Reconstruction ===")
for clip_num in range(1, 5):
    clip = f"clip{clip_num}"
    d = f"iclr_video/32x video results/{clip}"
    
    ltx_file = f"{d}/ltx_4.mp4" if clip_num == 4 else f"{d}/ltx.mp4"
    
    merge_videos(
        inputs=[f"{d}/groundtruth.mp4", f"{d}/ARVAE(ours).mp4", ltx_file],
        labels=["Ground Truth", "ARVAE (Ours)", "LTX"],
        output_name=f"32x_{clip}.mp4"
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
    output_name="gen_sky.mp4"
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
    output_name="gen_taichi.mp4"
)

print("\n=== Zero-Out Test ===")
merge_videos(
    inputs=[
        "1_zero_out_test/videos/8x_ch4/videoSRC05/normal.mp4",
        "1_zero_out_test/videos/8x_ch4/videoSRC05/zero_temporal.mp4",
        "1_zero_out_test/videos/8x_ch4/videoSRC05/zero_spatial.mp4",
    ],
    labels=["Normal", "Zero Temporal", "Zero Spatial"],
    output_name="zero_out_test.mp4"
)

print("\n=== Latent Swap Test ===")
merge_videos(
    inputs=[
        "2_latent_swap_test/videos/8x_ch4/swap_temporal.mp4",
        "2_latent_swap_test/videos/8x_ch4/swap_spatial.mp4",
    ],
    labels=["Swap Temporal", "Swap Spatial"],
    output_name="latent_swap_test.mp4"
)

print("\n=== DONE ===")
print(f"Merged videos saved to: {OUT_DIR}")
# List output files
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".mp4"):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}: {sz/1024:.0f} KB")
