from fastapi import FastAPI, File, UploadFile
from SCRIPTS.TRAIN.TEST import run_inference
from pathlib import Path
from pydantic import BaseModel
import shutil

app = FastAPI()

model_path = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\MODEL\SPEECH_TO_TEXT_MODEL_TRAINED.pth")
vocab_path = Path(r"C:\Users\MyLaptopKart\Desktop\Speech_to_Text_AI\vocab.json")


@app.post("/caption")
async def captions(audio_file: UploadFile = File(...)):
    # Specify a temporary path to save the uploaded file
    temp_file_path = Path("temp_audio_file.wav")

    # Save the uploaded file to the temporary location
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Run inference on the saved audio file
    caption = run_inference(audio_file_path=temp_file_path, model_path=model_path, vocab_path=vocab_path)

    # Clean up the temporary file after processing
    temp_file_path.unlink()

    # Return the generated caption
    return {"captions": caption}