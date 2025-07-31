import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np
import torch
import os
import subprocess
from pydub import AudioSegment, silence
import time


# --- Configuration (can be moved to a config file) ---
CONFIG = {
    "output_dir": "transcribed_videos",
    "temp_dir": "temp",
    "asr_model": "whisper_large_v3", # or "sarvam_ai_api", "google_cloud" etc.
    "translation_model": "indic_mt_model", # or "google_translate_api"
    "target_languages": ["hi", "bn", "ta"], # Example: Hindi, Bengali, Tamil
    "silence_threshold": -40, # dBFS, adjust based on audio
    "min_silence_len": 1000, # milliseconds
    "padding": 500, # milliseconds of padding around segments
    "font_path": "path/to/your/font.ttf", # Path to a TrueType Font file
    "font_size": 40,
    "font_color": "white",
    "text_position": ("center", "bottom"), # (x, y) or ("center", "bottom")
    "stroke_color": "black",
    "stroke_width": 2,
}



class VideoTranscriber:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["temp_dir"], exist_ok=True) 
        try:
            self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
            self.model.eval()  # Set to evaluation mode
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)    
            print(f"Model loaded and moved to: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have an internet connection and the model name is correct.")
            print("You might need to install 'transformers' if you encounter errors related to it.")
            exit()

    def extract_audio(self, video_path):        
        audio_path = os.path.join(self.config["temp_dir"], "audio.wav")
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", audio_path
        ]
        subprocess.run(command, check=True)
        return audio_path

    def segment_audio_on_silence(self, audio_path):
        audio = AudioSegment.from_wav(audio_path)
        segments = silence.split_on_silence(
            audio,
            min_silence_len=self.config["min_silence_len"],
            silence_thresh=self.config["silence_threshold"],
            keep_silence=self.config["padding"]
        )
        
        # Store segments with start/end times relative to original audio
        segmented_audio_data = []
        current_time_ms = 0
        for i, segment in enumerate(segments):
            segment_path = os.path.join(self.config["temp_dir"], f"segment_{i}.wav")
            segment.export(segment_path, format="wav")
            segmented_audio_data.append({
                "path": segment_path,
                "start_time_ms": current_time_ms,
                "end_time_ms": current_time_ms + len(segment)
            })
            current_time_ms += len(segment)
        return segmented_audio_data


    def transcribe_audio_segments(self, audio_segment_path, source_language='en'):
        segments, info = self.model.transcribe(audio_segment_path, language=source_language)
        transcribed_text = " ".join([seg.text for seg in segments])
        return transcribed_text 
    
       
         


if __name__ == "__main__":
    test_video_path = "/home/pgaikwad/asr/video_transcriber/english_video.mp4" 
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(test_video_path):
        print(f"Test video '{test_video_path}' does not exist. Please provide a valid video file.")
        exit()
        
    transcriber = VideoTranscriber(CONFIG)
    audio = transcriber.extract_audio(test_video_path)
    segments = transcriber.segment_audio_on_silence(audio)
    #transcription = transcriber.transcribe_audio_segments(segments[0]['path'], source_language='en')
    
    transcription_results = []
    for segment_info in segments:
        transcription = transcriber.transcribe_audio_segments(segment_info["path"])
        segment_info["transcription"] = transcription
        transcription_results.append(segment_info)

    
   
