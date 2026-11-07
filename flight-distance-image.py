import os
import cv2
import numpy as np

def extract_flight_distance(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    chart_box = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            x, y, w, h = cv2.boundingRect(c)
            if h > 100 and w > 100: 
                max_area = area
                chart_box = (y, y + h) 

    if chart_box is None:
        print(f"Could not auto-detect chart grid for {image_path}")
        return None

    pixel_y_at_500, pixel_y_at_0 = chart_box

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 220])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    path_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not path_contours:
        print(f"No flight path detected in {image_path}")
        return None

    main_contour = max(path_contours, key=cv2.contourArea)
    y_coords = main_contour[:, :, 1]
    min_y = np.min(y_coords) 

    total_height_pixels = pixel_y_at_0 - pixel_y_at_500
    if total_height_pixels <= 0:
        return None
        
    pixels_from_zero = pixel_y_at_0 - min_y
    estimated_distance = (pixels_from_zero / total_height_pixels) * 500
    
    return round(estimated_distance, 1)

def batch_process_directory(input_dir):
    print(f"Processing images in: {input_dir}\n")
    results = {}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(input_dir, filename)
            distance = extract_flight_distance(path)
            if distance is not None:
                results[filename] = distance
                print(f"{filename}: ~{distance} feet")
                
    return results

batch_process_directory("images")