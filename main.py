import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from processingVideo import process_video
from add_missing_data import process_interpolation
from visualize import video_output
from processCsv import final_csv

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Ensure upload and output directories exist
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
RESULTS_FOLDER = "frontend/static/results"

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

final_output_csv = os.path.join(OUTPUT_FOLDER, "final_csv.csv").replace("\\", "/")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    # Ensure the filename is safe
    filename = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded video file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    file_path = file_path.replace("\\", "/")

    return {"message": "Video uploaded successfully!", "file_path": file_path}

@app.post("/process_video/")
async def process_uploaded_video(file: UploadFile = File(...)):
    # Ensure the filename is safe
    filename = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, filename).replace("\\", "/")
    output_video_path = os.path.join(RESULTS_FOLDER, filename).replace("\\", "/")
    output_test_file = os.path.join(OUTPUT_FOLDER, "test.csv").replace("\\", "/")
    output_interpolated_file = os.path.join(OUTPUT_FOLDER, "test_interpolated.csv").replace("\\", "/")
    

    # Save the uploaded video file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Step 1: Process video frames
    results = process_video(file_path, output_csv=output_test_file)
    
    # Step 2: Process video frames for missing data
    process_interpolation(results, output_csv=output_interpolated_file)
    
    # Step 3: Generate video output with detected plates and interpolated data
    
    video_output(file_path, output_video_path, output_interpolated_file)
    
    # Step 4: Generate final csv with interpolated data
    res = final_csv(output_interpolated_file)
    
    return {
        "message": "Processing completed", 
        "data": res, 
        "processed_video": output_video_path
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)