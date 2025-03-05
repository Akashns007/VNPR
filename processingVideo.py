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
                    
                    if license_plate_text is not None:
                        results[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
    return results

def process_video(video_path, output_csv):
    """Process the entire video frame by frame."""
    cap = cv2.VideoCapture(video_path)
    coco_model, license_plate_detector, mot_tracker = initialize_models()
    vehicles = [2, 3, 5, 7]  # COCO dataset vehicle class IDs
    
    frame_nmr = -1
    results = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_nmr += 1
        frame_results = process_frame(frame, coco_model, license_plate_detector, mot_tracker, vehicles)
        if frame_results:
            results[frame_nmr] = frame_results
    
    cap.release()
    print(results)
    if results:
        write_csv(results, output_csv)
        return output_csv
    else:
        return None


if __name__ == '__main__':
    video_path = 'uploads/sample1.mp4'
    output_csv = 'output/test.csv'
    process_video(video_path, output_csv)