# image_processing.py

import cv2
import os
import numpy as np
import math

def process_image(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Dimensions of the image
    image_width = image.shape[1]
    image_height = image.shape[0]

    # Convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw the contours on
    image_with_contours = image.copy()

    # Draw all contours on the image
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw in green with thickness of 2

    image_with_boxes = image.copy()
    black_screen = np.zeros_like(image)

    # Create a list to hold the bounding boxes that pass the area threshold
    bounding_boxes = []

    # Loop through each contour to filter based on area
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the area of the bounding box
        area = w * h

        # Filter out very small boxes (area threshold can be adjusted)
        if area > 25000:  # Adjust this threshold as needed
            bounding_boxes.append((x, y, w, h))

            # Draw the bounding box on both images
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(black_screen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Sort the remaining bounding boxes by area in descending order
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[2] * box[3], reverse=True)

    # Get the largest bounding box
    largest_box = bounding_boxes[0]
    x, y, w, h = largest_box

    # Create a blank image to visualize the largest bounding box
    output_image = np.zeros_like(image)

    # Draw the largest bounding box on the blank image
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Label the dimensions on the image
    label = f'Width: {w}px, Height: {h}px'
    cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # Save the processed image in the 'uploads' folder in the root directory
    processed_image_filename = 'processed_' + os.path.basename(image_path)
    processed_image_path = os.path.join(os.getcwd(), 'uploads', processed_image_filename)

    # Save the processed image
    cv2.imwrite(processed_image_path, image_with_boxes)

    # Dimensions
    pixel_width = w
    pixel_height = h

    return processed_image_filename, image_width, image_height, pixel_width, pixel_height, largest_box




def real_dimensions(sensor_length, sensor_width, focal_length, altitude, image_width, image_height, pixel_width, pixel_height):
    sensor_length = int(sensor_length)
    sensor_width = int(sensor_width)
    focal_length = int(focal_length)
    image_width = int(image_width)
    image_height = int(image_height)
    altitude = int(altitude)
    pixel_width = int(pixel_width)
    pixel_height = int(pixel_height)

    # Calculate GSD (Ground Sampling Distance) for this drone
    gsd_m_per_px_l = (sensor_length * altitude) / (focal_length * image_height)
    gsd_m_per_px_w = (sensor_width * altitude) / (focal_length * image_width)

    # Convert the pixel dimensions to meters
    width_m = pixel_width * gsd_m_per_px_w
    height_m = pixel_height * gsd_m_per_px_l

    # Area and perimeter
    perimeter_m = 2*(width_m + height_m)
    area_m = width_m*height_m

    return width_m, height_m, perimeter_m, area_m, gsd_m_per_px_l, gsd_m_per_px_w


def point_to_bounding_box_distance(side, point, bounding_box, gsd_l, gsd_w):
    """
    Calculate the minimum distance between a point (x, y) and the bounding box.
    The bounding box is represented by [x1, y1, x2, y2].
    """
    x, y = point
    x1, y1, x2, y2 = bounding_box
    x2 = x2+x1
    y2 = y2+y1

    if(side=='side_1'):
        return (math.sqrt((y1-y)**2) * gsd_l)
    if(side=='side_2'):
        return (math.sqrt((x2-x)**2) * gsd_w)
    if(side=='side_3'):
        return (math.sqrt((y2-y)**2) * gsd_l)
    if(side=='side_4'):
        return (math.sqrt((x1-x)**2) * gsd_w)
    # Horizontal distance (left or right edge of the bounding box)
    #dx = min(abs(x - x1), abs(x - x2))

    # Vertical distance (top or bottom edge of the bounding box)
    #dy = min(abs(y - y1), abs(y - y2))

    # The minimum distance to any edge of the bounding box is the smaller of dx or dy
    #return math.sqrt(dx**2 + dy**2)


def calculate_distance(point1, point2, gsd_l, gsd_w, axis):
    """
    Calculate the Euclidean distance between two points.
    Points are in the form (x, y).
    """
    x1, y1 = point1
    x2, y2 = point2
    if axis=='x':
        return (math.sqrt((x2 - x1)**2 + (y2 - y1)**2)*gsd_w)
    elif axis=='y':
        return (math.sqrt((x2 - x1)**2 + (y2 - y1)**2)*gsd_l)



def detect_bounding_boxes(predicted_mask_image, image_path, width, height):
    # Read image using OpenCV
    #image = predicted_mask_image
    image = cv2.imread(image_path)

    curr_width = image.shape[1]
    curr_height = image.shape[0]

    # Resize the image
    resized_image = cv2.resize(image, (width, height))

    # Calculate scaling factors
    scale_w = width / curr_width
    scale_h = height / curr_height

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to hold the bounding boxes that pass the area threshold
    bounding_boxes = []

     # Loop through each contour to filter based on area
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the area of the bounding box
        area = w * h

        # Filter out very small boxes (area threshold can be adjusted)
        if area > 100:  # Adjust this threshold as needed
            bounding_boxes.append((x, y, w, h))

    # Adjust bounding boxes for the resized image
    resized_bounding_boxes = []

    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        # Scale the bounding box coordinates and dimensions
        new_x = int(x * scale_w)
        new_y = int(y * scale_h)
        new_w = int(w * scale_w)
        new_h = int(h * scale_h)

        # Store the resized bounding box
        resized_bounding_boxes.append((new_x, new_y, new_w, new_h))

        # Draw the resized bounding box on the resized image
        cv2.rectangle(resized_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 10)

        label = str(idx)
        cv2.putText(resized_image, label, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    # Save the processed image
    cv2.imwrite(image_path, resized_image)

    return resized_bounding_boxes