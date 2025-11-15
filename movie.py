import json
import re
import subprocess

from moviepy.editor import *


def main():
    # SadTalker command
    command = [
        "python",
        "inference.py",
        "--driven_audio",
        "out.wav",
        "--source_image",
        "Smiling Teacher by the Chalkboard.png",
        "--preprocess",
        "full",
        "--still",
    ]

    # Run process and capture stdout
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    output_lines = []
    for line in process.stdout:
        print(line, end="")  # Show real-time SadTalker output
        output_lines.append(line)

    process.wait()

    # Regex to extract all generated video names from output
    video_regex = re.compile(r"The generated video is named[: ]+(.+\.mp4)")
    video_files = []
    for line in output_lines:
        match = video_regex.search(line)
        if match:
            video_files.append(match.group(1).strip())

    # Get the final generated video (same line as in your sample output)
    if video_files:
        final_video = video_files[-1]  # Last one is usually the best (full resolution)
        print(f"\nExtracted SadTalker video: {final_video}")
    else:
        print("❌ No video file found in SadTalker output.")
    # Load your SadTalker output video (talking teacher)
    base_clip = VideoFileClip(final_video).resize(height=1080)

    # Example blackboard area (adjust values as needed)
    blackboard_box = (81, 92)  # (x, y) location of blackboard top-left
    blackboard_size = (760, 444)  # (width, height) of blackboard

    # Load timeline
    with open("video_timeline_with_images.json", "r") as f:
        timeline = json.load(f)

    # Create overlays for timeline images
    overlays = []
    for segment in timeline:
        overlay_img = (
            ImageClip(segment["image_path"])
            .set_duration(segment["end_time"] - segment["start_time"])
            .set_start(segment["start_time"])
            .resize(blackboard_size)
            .set_position(blackboard_box)
        )
        if segment.get("transition") == "fade":
            overlay_img = overlay_img.crossfadein(1.0).crossfadeout(1.0)
        overlays.append(overlay_img)

    # Composite overlays onto base (talking video)
    final_video = CompositeVideoClip([base_clip] + overlays, size=base_clip.size)

    # Export final video
    final_video.write_videofile("out.mp4", fps=30, codec="libx264")


if __name__ == "__main__":
    main()
