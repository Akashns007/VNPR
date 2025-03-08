import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import requests
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'backend/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Update the processed videos folder to match your actual storage location
STATIC_RESULTS_FOLDER = 'static/results'
os.makedirs(STATIC_RESULTS_FOLDER, exist_ok=True)

FASTAPI_URL = "http://localhost:8000/process_video/"
CSV_PATH = "../output/final_csv.csv" 

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    """Route to stream video from FastAPI."""
    return redirect("http://localhost:8000/video_feed/")

@app.route('/static/results/<path:filename>')
def serve_video(filename):
    """Serve video files from the static/results directory with proper MIME type."""
    # Determine the MIME type based on file extension
    mime_type = 'video/mp4' if filename.endswith('.mp4') else 'video/x-msvideo'
    return send_from_directory(STATIC_RESULTS_FOLDER, filename, mimetype=mime_type)

@app.route('/final_results')
def final_results():
    """Load and display results from final_csv.csv."""
    if not os.path.exists(CSV_PATH):
        return "No results available yet."

    df = pd.read_csv(CSV_PATH)
    return render_template('final_results.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = file.filename.replace(" ", "_")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Make sure data is a dictionary, not a string
            data_dict = response_data.get('data', {})
            if isinstance(data_dict, str):
                data_dict = {}
            
            # Get the video URL from the response
            video_url = response_data.get('processed_video', '')
            
            # Check if the video exists and handle path conversion if needed
            video_filename = os.path.basename(video_url)
            
            # Create a direct URL to the video file in static/results
            if video_filename:
                video_url = url_for('serve_video', filename=video_filename)
                
            return render_template('results.html', 
                                   video_url=video_url, 
                                   data=data_dict,
                                   filename=video_filename)
        else:
            return f"Error processing video: {response.text}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)