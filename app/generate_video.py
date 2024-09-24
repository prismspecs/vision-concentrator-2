from generate_images import generate_images
from moviepy.editor import ImageSequenceClip
import os

def create_vision_video(visions):
    image_paths = generate_images(visions)
    fps = 15  # Frames per second for the video

    print("Creating video from generated frames...")
    clip = ImageSequenceClip(image_paths, fps=fps)
    os.makedirs('static', exist_ok=True)
    video_path = os.path.join('static', 'vision_video.mp4')
    clip.write_videofile(video_path, codec='libx264')
    return video_path
