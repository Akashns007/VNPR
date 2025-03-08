import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """Draws a stylized border around detected objects."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def load_results(csv_path):
    """Loads detection results from CSV."""
    return pd.read_csv(csv_path)

# def initialize_video(video_path, output_path):
#     """Initializes video input and output settings."""
#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     return cap, out, fps, width, height

def initialize_video(video_path, output_path):
    """Initializes video input and output settings."""
    cap = cv2.VideoCapture(video_path)
    
    # Use H.264 codec for better web compatibility
    # 'avc1' is the FourCC code for H.264
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, out, fps, width, height

def extract_best_license_plates(cap, results):
    """Extracts the best license plate for each detected vehicle."""
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        best_plate_data = results[(results['car_id'] == car_id) & 
                                  (results['license_number_score'] == max_score)].iloc[0]
        
        plate_number = best_plate_data['license_number']
        frame_number = best_plate_data['frame_nmr']
        bbox_str = best_plate_data['license_plate_bbox']

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            x1, y1, x2, y2 = ast.literal_eval(bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

            license_plate[car_id] = {'license_crop': license_crop, 'license_plate_number': plate_number}

    return license_plate

def overlay_license_plate(frame, license_crop, plate_number, car_x1, car_x2, car_y1, H):
    """Overlays the license plate crop and text on the frame."""
    W = license_crop.shape[1]
    
    try:
        frame[int(car_y1) - H - 100:int(car_y1) - 100,
              int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

        frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
              int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

        (text_width, text_height), _ = cv2.getTextSize(plate_number, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)

        cv2.putText(frame, plate_number,
                    (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
    except:
        pass

def process_video(cap, out, results, license_plate):
    """Processes the video frame by frame, drawing bounding boxes and overlaying license plates."""
    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True

    while ret:
        ret, frame = cap.read()
        frame_nmr += 1

        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for _, row in df_.iterrows():
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)

                x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                license_crop = license_plate[row['car_id']]['license_crop']
                H = license_crop.shape[0]
                overlay_license_plate(frame, license_crop, license_plate[row['car_id']]['license_plate_number'], car_x1, car_x2, car_y1, H)

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

    out.release()
    cap.release()

def video_output(video_path, output_path, csv_path):
    """function to execute the vehicle license plate detection pipeline."""
    # csv_path = 'output/test_interpolated.csv'
    # video_path = 'sample1.mp4'
    # output_path = './out.mp4'

    results = load_results(csv_path)
    cap, out, fps, width, height = initialize_video(video_path, output_path)
    license_plate = extract_best_license_plates(cap, results)
    process_video(cap, out, results, license_plate)

if __name__ == "__main__":
    video_output("uploads/sample1.mp4", "frontend/static/results/sample1.mp4", "output/test_interpolated.csv")
