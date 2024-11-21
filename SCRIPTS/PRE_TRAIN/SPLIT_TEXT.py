import re
import os

def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.replace(',','.').split(":")
    return int(hours)*3600 + int(minutes)*60 + float(seconds)

text_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\TEXT_FILES"
processed_audio_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\PROCESSED_AUDIO"
split_text_directory = r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\SPLIT_TEXT"
if not os.path.exists(split_text_directory):
    os.makedirs(split_text_directory)
  
# splitting text file on the basis of audio segment files
def split_text_file(text_file, processed_audio_directory, split_text_directory):
    with open(text_file, 'r') as f:
        lines = f.readlines()

    text_segments = {} 
    segment_count = 0
    skipped_segments = 0

# finding the time stamps written for naming and saving it in diff file
    i = 0
    while i < len(lines):
        start_line = lines[i].strip()
        if not re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', start_line):
            i = i + 1
            continue

        start_time_str, end_time_str = start_line.split("-->")
        start_time = time_to_seconds(start_time_str)
        end_time = time_to_seconds(end_time_str)

        i = i + 1
        text_lines = []

# storing the text relating to the time stamps for saving
        while i<len(lines) and not re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', lines[i]):
            text_lines.append(lines[i].strip())
            i = i + 1

        text = ' '.join(text_lines)

        segment_key = f"{start_time:.2f}_to_{end_time:.2f}".replace(".","_")
        text_segments[segment_key] = text
        segment_count += 1

    original_base_name = os.path.splitext(os.path.basename(text_file))[0]

# checking and naming text file same as the audio file
    for segment, text in text_segments.items():
        segment_file_name = f"{original_base_name.replace(' ','_')}_{segment}.txt"
        segment_file_path = os.path.join(split_text_directory,segment_file_name)

        expected_audio_filename =  f"{original_base_name.replace(' ','_')}_{segment}.wav"
        matching_audio_file_path = os.path.join(processed_audio_directory,expected_audio_filename)

# writing and saving the splitted text files
        if os.path.exists(matching_audio_file_path):
            with open(segment_file_path, 'w') as f:
                f.write(f"Start: {segment.split('_to_')[0].replace('_','.')}s\n")
                f.write(f"End: {segment.split('_to_')[1].replace('_','.')}s\n")
                f.write(f"Text: {text}\n")

            print("")
            print(f"Segment file saved: {segment_file_path}")
        else:
            print(f"No matching audio file found for segment {segment}. Skipping.")
            skipped_segments += 1

    print(f"Processed {segment_count} segments; {skipped_segments} segments skipped.")


text_files = [f for f in os.listdir(text_directory) if f.lower().endswith(".txt")]
if not text_files:
    raise FileNotFoundError("No text transcription files found in Text directory.")

for text_file_name in text_files:
    text_file_path = os.path.join(text_directory,text_file_name)
    if os.path.exists(text_file_path) and os.access(text_file_path, os.R_OK):
        print(f"Processing Text File: {text_file_path}")
        try:
            split_text_file(text_file_path,processed_audio_directory,split_text_directory)
        except Exception as e :
            print(f"Error in processing {text_file_name}: {e}")

    else:
        print(f"Text file {text_file_path} is not accessible.")
