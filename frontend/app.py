import os
from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

UPLOAD_FOLDER = 'backend/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FASTAPI_URL = "http://localhost:8000/process_video/"

@app.route('/')
def index():
    return render_template('index.html')

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
            data = response.json()
            return render_template('results.html', 
                                   video_url=data.get('processed_video', ''), 
                                   data=data.get('data', {}))
        else:
            return f"Error processing video: {response.text}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)