from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

def initialize_models():
    """Load the YOLO models and initialize the tracker."""
    coco_model = YOLO('models/yolov8n.pt').to("cuda")
    license_plate_detector = YOLO('models/license_plate_detector.pt').to("cuda")
    mot_tracker = Sort()
    return coco_model, license_plate_detector, mot_tracker

def classify_license_plate_color(license_plate_crop):
    """Classify the license plate color into one of the four categories."""
    hsv = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))
    h, s, v = avg_color
    
    if 35 <= h <= 85 and s > 50 and v > 50:
        return "Green"
    elif 20 <= h <= 35 and s > 50 and v > 50:
        return "Yellow"
    elif v > 150 and s < 50:
        return "White"
    elif 20 <= h <= 35 and s > 50 and v > 50 and np.mean(hsv[:, :, 2]) < 100:
        return "Yellow text on Green"
    else:
        return "Unknown"

def process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicles):
    """Process a single frame for vehicle and license plate detection."""
    results = {}
    
    # Detect vehicles
    detections = coco_model(frame, stream=True)
    detections = next(detections, None)
    
    detections_ = []
    if detections and detections.boxes is not None:
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
    
    # Track vehicles
    # Track vehicles
    if len(detections_) == 0:
        track_ids = np.empty((0, 5))  # Prevent empty input error
    else:
        track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame, stream=True)
    license_plates = next(license_plates, None)
    
    if license_plates and license_plates.boxes is not None:
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Assign license plate to a detected car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                if license_plate_crop is not None and license_plate_crop.size > 0:
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    
                    # Determine license plate color
                    license_plate_color = classify_license_plate_color(license_plate_crop)
                    
                    if license_plate_text is not None:
                        results[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score,
                                'color': license_plate_color
                            }
                        }
    return results
        
import csv

def generate_frames(output_csv="output/live_results.csv"):
    """Generate frames from webcam for live streaming and save detections to CSV."""
    coco_model, license_plate_detector, mot_tracker = initialize_models()
    vehicles = [2, 3, 5, 7]  # COCO dataset vehicle class IDs

    cap = cv2.VideoCapture(0)  # Use webcam

    # Prepare CSV file
    with open(output_csv, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_nmr", "car_id", "car_bbox", "license_plate_bbox", 
                         "license_plate_bbox_score", "license_number", 
                         "license_number_score", "license_plate_color"])

        frame_nmr = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_results = process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicles)

            # Save results to CSV if license plates are detected
            if frame_results:
                for car_id, data in frame_results.items():
                    if "car" in data and "license_plate" in data:
                        writer.writerow([
                            frame_nmr,
                            car_id,
                            data["car"]["bbox"],
                            data["license_plate"]["bbox"],
                            data["license_plate"]["bbox_score"],
                            data["license_plate"]["text"],
                            data["license_plate"]["text_score"],
                            data["license_plate"]["color"]
                        ])

            frame_nmr += 1

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    
def process_input(input_source, output_csv=None):
    """Process a video, image, or live feed dynamically."""
    coco_model, license_plate_detector, mot_tracker = initialize_models()
    vehicles = [2, 3, 5, 7]  # COCO dataset vehicle class IDs
    
    cap = cv2.VideoCapture(input_source)
    
    frame_nmr = -1
    results = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_nmr += 1
        frame_results = process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicles)
        if frame_results:
            results[frame_nmr] = frame_results  # Merge instead of nesting under frame number
    
    cap.release()
    cv2.destroyAllWindows()
    
    if results:
        write_csv(results, output_csv)
        return output_csv
    else:
        return None

if __name__ == '__main__':
    input_source = "uploads/sample_small.mp4"  # Change to 0 for webcam or provide image path
    output_csv = 'output/test.csv'
    print(process_input(input_source, output_csv))
