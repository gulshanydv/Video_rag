import whisper
import openai
from moviepy import VideoFileClip
import os

# 1. Extract audio
def extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

# 2. Transcribe using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


# --- Usage ---
video_folder = "/home/anviam97/Documents/Video_Rag/vidoes"  # Put all your .mp4 files here
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(video_folder):
    if filename.lower().endswith(".mp4"):
        video_path = os.path.join(video_folder, filename)
        audio_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.mp3")
        
        print(f"\nProcessing: {filename}")
        extract_audio(video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        
        print("Transcript:\n", transcript, "...\n")  # Show preview
        with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_transcript.txt"), "w") as f:
            f.write(transcript)
        
