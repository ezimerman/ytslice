import os
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

import subprocess
import json

from typing import List

# load env vars
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

youtube_url = "https://www.youtube.com/watch?v=qkYoBNdcXBU"

os.makedirs("downloaded_videos", exist_ok=True)

# pytube download
yt = YouTube(youtube_url)
video = yt.streams.filter(file_extension='mp4').first()
safe_title = yt.title.replace(' ', '_')
filename = f"downloaded_videos/{safe_title}.mp4"

video.download(filename=filename)

#get transcript
video_id = yt.video_id
transcript = YouTubeTranscriptApi.get_transcript(video_id)

# define llm
llm = ChatOpenAI(model='gpt-4o', temperature=0.7, max_tokens=None, timeout=None, max_retries=2)


# build llm prompt
prompt = f"""Provided to you is a transcript of a video. Please identify all segments that can be extracted as subtopics from the video based on the transcript. Make sure each segment is between 30-500 seconds in duration. Make sure you provide extremely accurate timestamps and respond only in the format provided.
\n Here is the transcription : \n {transcript}"""

messages = [
    {"role": "system", "content": "You are a business analyst. You are a master at reading youtube transcripts and understanding the context of complex health insurance industry business processes. You have extraordinary skills to extract subtopic content. Your subtopics can be repurposed as separate videos."},
    {"role": "user", "content": prompt}
]

class Segment(BaseModel):
    # represent a video segment
    start_time: float = Field(..., description="The start time in seconds")
    end_time: float = Field(..., description="The end time in seconds")
    yt_title: str = Field(..., description="The video title to clearly understand the subtopic")
    description: str = Field(..., description="The detailed video description to provide a clear summary")
    duration: int = Field(..., description="The duration of the segmnent in seconds")

class VideoTranscript(BaseModel):
    # represent transcript of video segments
    segments: List[Segment] = Field(..., description="List of viral segments in the video")


structured_llm = llm.with_structured_output(VideoTranscript)
ai_msg = structured_llm.invoke(messages)
parsed_content = ai_msg.dict()['segments']

# create folder for clips
os.makedirs(f"generated_clips", exist_ok=True)
segment_labels = []
video_title = safe_title

for i, segment in enumerate(parsed_content):
    start_time = segment['start_time']
    end_time = segment['end_time']
    yt_title = segment['yt_title']
    description = segment['description']
    duration = segment['duration']

    output_file = f"generated_clips/{video_title}_{str(i)}.mp4"
    command = f"ffmpeg -i {filename} -ss {start_time} -to {end_time} -c:v libx264 -c:a aac -strict experimental -b:a 192k {output_file}"
    subprocess.call(command, shell=True)
    segment_labels.append(f"Sub-Topic {i+1}: {yt_title}, Duration: {duration}s\nDescription: {description}\n")

with open('generated_clips/segment_labels.txt', 'w') as f:
    for label in segment_labels:
        f.write(label +"\n")

# save segments to json
with open('generated_clips/segments.json', 'w') as f:
    json.dump(parsed_content, f, indent=4)

