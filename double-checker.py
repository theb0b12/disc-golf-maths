import cv2
import numpy as np

def calculate_distanceNN(speed, glide, turn, fade):
    # Your optimized solution vector (no bias)
    x1 = 16.3389   # Speed multiplier
    x2 = 38.3866   # Glide multiplier
    x3 = -4.4266  # Turn multiplier
    x4 = 5.3333   # Fade multiplier
    x5 = 12.6581
    
    # Equation: Distance = (x1 * Speed) + (x2 * Glide) + (x3 * Turn) + (x4 * Fade) + bias
    distance = (x1 * speed) + (x2 * glide) + (x3 * turn) + (x4 * fade) + x5
    return round(distance, 1)

def calculate_distanceAUG(speed, glide, turn, fade):
    # Your optimized solution vector (no bias)
    x1 = 15.4333   # Speed multiplier
    x2 = 38.8389   # Glide multiplier
    x3 = -21.8056  # Turn multiplier
    x4 = 21.8722   # Fade multiplier
    
    # Equation: Distance = (x1 * Speed) + (x2 * Glide) + (x3 * Turn) + (x4 * Fade)
    distance = (x1 * speed) + (x2 * glide) + (x3 * turn) + (x4 * fade)
    return round(distance, 1)

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


def main():
    print("\nEnter path to image")
    image = input("   Path:   ")
    print("\nEnter the flight numbers:")
    speed = float(input("  Speed: "))
    glide = float(input("  Glide: "))
    turn = float(input("  Turn:  "))
    fade = float(input("  Fade:  "))


    total_distAUG = calculate_distanceAUG(speed, glide, turn, fade)
    total_distNN = calculate_distanceNN(speed, glide, turn, fade)
    real_total_dist = extract_flight_distance(image)

    print(f"Real Total Distance: {real_total_dist} feet")
    print(f"\nEstimated Total Distance with Augmented Matrix: {total_distAUG} feet")
    print(f"Estimated Total Distance with NN: {total_distNN} feet")

        

if __name__ == "__main__":
    main()