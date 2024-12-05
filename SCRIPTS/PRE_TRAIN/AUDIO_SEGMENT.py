"""
Audio Segmentation Script

This script segments audio files (.wav/.mp3) based on timestamps provided in text files (.txt). 
It extracts and saves smaller audio clips using the specified time ranges.

Main Functions:
- time_to_milliseconds: Converts 'hh:mm:ss,ms' to milliseconds.
- get_output_filename: Creates filenames for audio segments.

Output:
Segmented audio clips saved in the 'PROCESSED_AUDIO' directory.
"""

import os
from pydub import AudioSegment

# Directories for audio input and text input
audio_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\AUDIO_FILES"
text_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\TEXT_FILES"

print ("Files in Audio Directory", os.listdir(audio_directory))
print ("Files in Text Directory: ", os.listdir(text_directory))

audio_files = [f for f in os.listdir(audio_directory) if f.lower().endswith(('.wav','.mp3'))]
if not audio_files:
    raise FileNotFoundError("NO files in Audio Directory.")

text_files = [f for f in os.listdir(text_directory) if f.lower().endswith('.txt')]
if not text_files:
    raise FileNotFoundError("NO text files in Text Directory.")

# Directory to store processed audio segments
processed_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\PROCESSED_AUDIO"
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

# Converts timestamp from 'hh:mm:ss,ms' format to milliseconds for precise slicing
def time_to_milliseconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    total_milliseconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)
    return total_milliseconds

# Generates the output filename using the base name and time range of the segment
def get_output_filename(base_name, start_time, end_time):
    base_name = base_name.replace(" ", "_")
    start_time_str = f"{start_time / 1000:.2f}".replace('.','_')
    end_time_str = f"{end_time / 1000:.2f}".replace('.', '_')
    return f"{base_name}_{start_time_str}_to_{end_time_str}.wav"

# Processing each audio file that has a matching text file
for audio_file in audio_files:
    audio_file_path = os.path.join(audio_directory, audio_file)
    print(f"Processing Audio File: {audio_file_path}")
    
    audio = AudioSegment.from_wav(audio_file_path)
    text_file_name = os.path.splitext(audio_file)[0] + '.txt'
    text_file_path = os.path.join(text_directory, text_file_name)

    with open(text_file_path, 'r') as f:
        lines = f.readlines()

    # Iterate over lines to extract timestamp ranges and corresponding text
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line: # Detect timestamp lines in the format "start --> end"
            times = line.split('-->')
            start_time = time_to_milliseconds(times[0].strip())
            end_time = time_to_milliseconds(times[1].strip())

            print(f"start time: {start_time}ms, end time: {end_time}ms")

            i += 1
            if i < len(lines):
                text_line = lines[i].strip() # Next line is the text for this segment
                print(f"Extracted text for segment: {text_line}")

                # Extract and save the audio segment using the specified time range
                audio_segment = audio[start_time:end_time]
                if len(audio_segment) == 0:
                    print(f"Empty audio segment from {start_time}ms to {end_time}ms, skipping.")
                    continue

                base_name = os.path.splitext(audio_file)[0]
                segment_file_name = get_output_filename(base_name, start_time, end_time)
                segment_path = os.path.join(processed_directory, segment_file_name)
                audio_segment.export(segment_path, format="wav")
                print(f"Segment saved: {segment_path}")
            i += 1
        else:
            i += 1