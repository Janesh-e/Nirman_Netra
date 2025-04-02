from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

import tensorflow as tf
import rasterio
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from PIL import Image


def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    # Add batch dimension (since the model expects input shape [batch_size, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_cd(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.reshape(1, target_size[0], target_size[1], 3)  # Add batch dimension


def preprocess_image_wd(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def convert_to_grayscale(image_path, save_path=None):
    try:
        img = Image.open(image_path)
        grayscale_img = img.convert("L")
        if save_path:
            grayscale_img.save(save_path)
        return grayscale_img
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def post_process_wmask(predicted_mask, output_path):
    #cv2.imwrite(output_path, predicted_mask)
    min_area=1500
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
    if len(predicted_mask.shape) > 2:
        predicted_mask = predicted_mask.squeeze()  # Remove batch dimension if present

    # Step 1: Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(predicted_mask, connectivity=8)
    # Create a clean mask retaining only large components
    clean_mask = np.zeros_like(predicted_mask, dtype=np.uint8)

    # Step 3: Extract contours for each connected component (excluding the background)
    for label in range(1, num_labels):  # Skip label 0 (background)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == label] = 255
 
    boundary_mask = np.zeros_like(clean_mask, dtype=np.uint8)

    # Step 1: Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    # Step 3: Extract contours for each connected component (excluding the background)
    for label in range(1, num_labels):  # Skip label 0 (background)
        component_mask = (labels == label).astype(np.uint8)

        # Find contours for the current connected component
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contour on the boundary mask
        cv2.drawContours(boundary_mask, contours, -1, (255), thickness=1)  # Use thickness=1 for boundary line

    # Generate boundary mask
    #contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the boundary mask
    #cv2.drawContours(boundary_mask, contours, -1, (255), thickness=1)

    # Save the boundary mask
    cv2.imwrite(output_path, boundary_mask)
    
    return boundary_mask


def post_process_bmask(predicted_mask, output_path):
    #cv2.imwrite(output_path, predicted_mask)
    
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    if len(predicted_mask.shape) > 2:
        predicted_mask = predicted_mask.squeeze()

    kernel = np.ones((5, 5), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Perform closing (dilation followed by erosion) to fill small holes
    closed_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_CLOSE, kernel)

    # Apply Gaussian blur for smoothing
    blurred_mask = cv2.GaussianBlur(closed_mask, (5, 5), 0)

    # Reapply thresholding after blur
    final_mask = (blurred_mask > 128).astype(np.uint8) * 255

    # Ensure mask is binary
    _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    cv2.imwrite(output_path, final_mask)

    return final_mask, bounding_boxes

def draw_bounding_boxes(image, bounding_boxes):
    """
    Draw bounding boxes on the image.

    Args:
        image (numpy.ndarray): Image to draw bounding boxes on.
        bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes as (x, y, w, h).
    """
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw index number
        cv2.putText(image, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def combine_and_visualize_masks(post_waterbody_mask, post_building_mask, output_path):

    # Ensure masks are binary and have the same shape
    assert post_waterbody_mask.shape == post_building_mask.shape, "Masks must have the same shape"
    waterbody_mask = (post_waterbody_mask > 0).astype(np.uint8)
    building_mask = (post_building_mask > 0).astype(np.uint8)

    # Combine masks using bitwise OR
    combined_mask = cv2.bitwise_or(waterbody_mask, building_mask)

    combined_mask = (combined_mask * 255).astype(np.uint8)

    # Save the combined mask
    output_path = output_path.rsplit('.', 1)[0] + '.jpg'
    cv2.imwrite(output_path, combined_mask)

    return combined_mask


def visualize_distances_and_combine_masks(building_mask, waterbody_boundary_mask, distances, building_centroids, waterbody_coords, output_path):
    """
    Combines the building and waterbody masks into one image and visualizes distances
    by drawing lines between building centroids and the nearest waterbody boundary points.

    Args:
        building_mask (np.ndarray): Binary mask for buildings (1 for buildings, 0 otherwise).
        waterbody_boundary_mask (np.ndarray): Binary mask for waterbody boundaries (1 for boundary, 0 otherwise).
        distances (list of tuples): List of (address, distance) tuples.
        building_centroids (list of tuples): List of (row, col) coordinates for building centroids.
        waterbody_coords (list of tuples): List of (row, col) coordinates for waterbody boundary points.
        output_path (str): Path to save the combined image.
    """
    # Create a blank color image for visualization
    height, width = building_mask.shape
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Add building mask to the blue channel
    combined_image[building_mask > 0] = [255, 0, 0]  # Blue for buildings

    # Add waterbody boundary mask to the red channel
    combined_image[waterbody_boundary_mask > 0] = [0, 0, 255]  # Red for waterbody boundaries

    # Draw lines between building centroids and their nearest waterbody boundary points
    for centroid, waterbody_point in zip(building_centroids, waterbody_coords):
        # Convert (row, col) to (x, y) for OpenCV
        building_point = (int(centroid[1]), int(centroid[0]))
        waterbody_point_cv = (int(waterbody_point[1]), int(waterbody_point[0]))

        # Draw the line
        cv2.line(combined_image, building_point, waterbody_point_cv, (0, 255, 0), 1)  # Green line for distance

        # Optionally, draw the centroid points for buildings and waterbody
        cv2.circle(combined_image, building_point, 3, (255, 255, 255), -1)  # White for building centroid
        cv2.circle(combined_image, waterbody_point_cv, 3, (255, 255, 255), -1)  # White for waterbody point

    # Save the resulting visualization
    output_path = output_path.rsplit('.', 1)[0] + '.jpg'
    cv2.imwrite(output_path, combined_image)





def calculate_building_to_waterbody_distances(building_mask, waterbody_boundary_mask, tif_path, output_visualization_path):
    def pixel_to_geo(pixel_coords, transform):
        row, col = pixel_coords
        lon, lat = transform * (col, row)
        return lat, lon

    def calculate_individual_centroids(mask):
        # Label connected components
        num_labels, labeled_mask = cv2.connectedComponents(mask.astype(np.uint8))

        centroids = []
        for label in range(1, num_labels):  # Exclude the background label (0)
            # Get coordinates of pixels in the component
            coords = np.column_stack(np.where(labeled_mask == label))
            # Compute the centroid
            centroid = np.mean(coords, axis=0)
            centroids.append(tuple(centroid))

        return centroids

    def coordinates_to_address(lat_long_pairs):
        addresses = []
        geolocator = Nominatim(user_agent="building_waterbody_detector")
        for (lat, long) in lat_long_pairs:
            try:
                location = geolocator.reverse((lat, long), exactly_one=True)
                if location:
                    addresses.append(location.address)
                else:
                    addresses.append("Address not found")
            except Exception as e:
                print(f"Geocoding error for ({lat}, {long}): {e}")
                addresses.append("Geocoding failed")
        return addresses

    # Read geospatial data from the TIFF file
    with rasterio.open(tif_path) as src:
        transform = src.transform  # Get the affine transformation

    # Calculate centroids for buildings
    centroids_pixel = calculate_individual_centroids(building_mask)

    # Get geographic coordinates of waterbody boundary pixels
    waterbody_coords = np.column_stack(np.where(waterbody_boundary_mask > 0))

    waterbody_geo = [pixel_to_geo((row, col), transform) for row, col in waterbody_coords]

    # Convert centroids to geographic coordinates
    centroids_geo = [pixel_to_geo((centroid[0], centroid[1]), transform) for centroid in centroids_pixel]

    # Reverse geocode building centroids to get addresses
    addresses = coordinates_to_address(centroids_geo)

    # Calculate distances from each building centroid to the nearest waterbody boundary
    results = []
    nearest_waterbody_coords = []
    for i, (centroid, address) in enumerate(zip(centroids_geo, addresses)):
        # Compute geodesic distances
        distances_to_boundary = [geodesic(centroid, wb).meters for wb in waterbody_geo]
        print(distances_to_boundary)
        min_distance = min(distances_to_boundary) if distances_to_boundary else float('inf')
        # Find the nearest boundary point in pixel coordinates
        nearest_boundary_idx = distances_to_boundary.index(min_distance)
        nearest_waterbody_coords.append(waterbody_coords[nearest_boundary_idx])
        results.append((address, min_distance))

        # Visualize the combined masks and distances
    visualize_distances_and_combine_masks(
        building_mask,
        waterbody_boundary_mask,
        results,
        centroids_pixel,
        nearest_waterbody_coords,
        output_visualization_path
    )

    return results







# new
def pixel_to_geo_reg(pixel_coords, transform):
    """
    Convert pixel coordinates to geographic coordinates using rasterio's transform.
    """
    row, col = pixel_coords
    lon, lat = transform * (col, row)  # Apply the affine transform
    return lat, lon


def calculate_real_dimensions_from_bounding_box(bounding_boxes, orthophoto_path):
    """
    Calculate real-world dimensions for buildings using bounding box coordinates.
    """
    with rasterio.open(orthophoto_path) as src:
        transform = src.transform  # Get the affine transformation

    building_dimensions = []

    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        # Calculate the pixel coordinates of the corners
        top_left = (y, x)
        top_right = (y, x + w)
        bottom_left = (y + h, x)
        bottom_right = (y + h, x + w)

        # Convert pixel coordinates to geographic coordinates
        top_left_geo = pixel_to_geo_reg(top_left, transform)
        top_right_geo = pixel_to_geo_reg(top_right, transform)
        bottom_left_geo = pixel_to_geo_reg(bottom_left, transform)
        bottom_right_geo = pixel_to_geo_reg(bottom_right, transform)

        # Calculate real-world dimensions using geodesic distances
        real_width = geodesic(top_left_geo, top_right_geo).meters
        real_height = geodesic(top_left_geo, bottom_left_geo).meters

        building_dimensions.append({
            "building_id": idx + 1,
            "top_left": top_left_geo,
            "top_right": top_right_geo,
            "bottom_left": bottom_left_geo,
            "bottom_right": bottom_right_geo,
            "real_width": real_width,
            "real_height": real_height
        })

    return building_dimensions
