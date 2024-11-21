import os
from pydub import AudioSegment

print("Current working directory", os.getcwd())
audio_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\AUDIO_FILES"
text_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\TEXT_FILES"
'''
print(audio_directory)
print ("Files in Audio Directory", os.listdir(audio_directory))
print(text_directory)
print ("Files in Text Directory: ", os.listdir(text_directory))
'''

audio_files = [f for f in os.listdir(audio_directory) if f.lower().endswith(('.wav','.mp3'))]
if not audio_files:
    raise FileNotFoundError("NO files in Audio Directory.")

text_files = [f for f in os.listdir(text_directory) if f.lower().endswith('.txt')]
if not text_files:
    raise FileNotFoundError("NO text files in Text Directory.")

processed_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\PROCESSED_AUDIO"
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

# Converting the time written in audio for naming of trimmed audio files
def time_to_milliseconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    total_milliseconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)
    return total_milliseconds

# Naming of trimmed audio files
def get_output_filename(base_name, start_time, end_time):
    base_name = base_name.replace(" ", "_")
    start_time_str = f"{start_time / 1000:.2f}".replace('.','_')
    end_time_str = f"{end_time / 1000:.2f}".replace('.', '_')
    return f"{base_name}_{start_time_str}_to_{end_time_str}.wav"

# Iterating over each audio file that has a corrsponding text file
for audio_file in audio_files:
    audio_file_path = os.path.join(audio_directory, audio_file)
    print(f"Processing Audio File: {audio_file_path}")
    audio = AudioSegment.from_wav(audio_file_path)
    text_file_name = os.path.splitext(audio_file)[0] + '.txt'
    text_file_path = os.path.join(text_directory, text_file_name)

    with open(text_file_path, 'r') as f:
        lines = f.readlines()

# taking out start and end time to trim audio
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            times = line.split('-->')
            start_time = time_to_milliseconds(times[0].strip())
            end_time = time_to_milliseconds(times[1].strip())

            print(f"start time: {start_time}ms, end time: {end_time}ms")

            i += 1
            if i < len(lines):
                text_line = lines[i].strip()
                print(f"Extracted text for segment: {text_line}")

# audio segmenting from start time to end time
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