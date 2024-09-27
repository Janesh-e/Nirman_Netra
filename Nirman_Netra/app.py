from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from models import db, Drone  # Import models and db
import json


import os
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np

from image_processing.image_processing import process_image, real_dimensions, detect_bounding_boxes, point_to_bounding_box_distance, calculate_distance
from image_processing.model_utils import preprocess_image
from image_processing.reverse_geocoding import save_processed_image_with_geotags, pixel_to_coordinates, coordinates_to_address

app = Flask(__name__)

# Set the secret key to some random bytes or any secret string
app.secret_key = '1234567890'  # Replace with a strong, random key

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///drone_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database with the app
db.init_app(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image formats
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load TensorFlow model (assuming your model is saved as 'model.h5')
#model = tf.keras.models.load_model('model.h5')
# Initialize a global variable to hold the model
model = None
model = tf.keras.models.load_model('models/building_detection.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check-authorization')
def check_authorization():
    return render_template('check_authorization.html')

@app.route('/check-regulations', methods=['GET', 'POST'])
def check_regulations():
    if request.method == 'POST':
        # Handle form submission for uploading the image, calculating dimensions, etc.
        return redirect(url_for('calculate_actual_dimensions_check'))
    else:
        # Render the initial check regulations page
        return render_template('check_regulationss.html')

@app.route('/add_drone', methods=['POST'])
def add_drone():
    if request.method == 'POST':
        return render_template('check_regulationss.html', show_popup=True)
    return render_template('check_regulationss.html', show_popup=False)

@app.route('/submit_drone', methods=['POST'])
def submit_drone():
    drone_name = request.form['drone_name']
    sensor_length = request.form['sensor_length']
    sensor_width = request.form['sensor_width']
    focal_length = request.form['focal_length']

    new_drone = Drone(drone_name, sensor_length, sensor_width, focal_length)
    db.session.add(new_drone)
    db.session.commit()

    return render_template('check_regulationss.html')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty part without filename
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['uploaded_image'] = filename
        return render_template('check_regulationss.html', uploaded_image=file.filename)

    return redirect(request.url)


@app.route('/detect/<filename>')
def detect_building_dimensions(filename):
    # Build the full path to the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if the image file exists before processing
    if not os.path.exists(image_path):
        return f"Error: File {filename} not found", 404

    # Process the image
    processed_image, image_width, image_height, pixel_width, pixel_height, largest_box = process_image(image_path)

    session['processed_image'] = processed_image
    session['image_width'] = image_width
    session['image_height'] = image_height
    session['pixel_width'] = pixel_width
    session['pixel_height'] = pixel_height
    session['largest_box'] = largest_box

    # Render the template to display the uploaded and processed images
    return render_template('check_regulationss.html', uploaded_image=filename, processed_image=processed_image,
                            image_width=image_width, image_height=image_height, 
                            pixel_width=pixel_width, pixel_height=pixel_height, largest_box=largest_box)


@app.route('/calculate_actual_dimensions_check', methods=['POST'])
def calculate_actual_dimensions_check():
    processed_image = session.get('processed_image')
    pixel_width = session.get('pixel_width')
    pixel_height = session.get('pixel_height')
    image_width = session.get('image_width')
    image_height = session.get('image_height')
    bounding_box_str = request.form.get('largest_box')

    # Deserialize it back into a Python list or tuple
    largest_box = json.loads(bounding_box_str)

    drones = Drone.query.all()
    return render_template('check_regulationss.html', actual_dimensions=True, drones=drones, processed_image=processed_image,
                           pixel_width=pixel_width, pixel_height=pixel_height,
                           image_width=image_width, image_height=image_height, largest_box=largest_box)
    


@app.route('/find_actual_dimensions', methods=['POST'])
def find_actual_dimensions():
    processed_image = session.get('processed_image')
    pixel_width = session.get('pixel_width')
    pixel_height = session.get('pixel_height')
    image_width = session.get('image_width')
    image_height = session.get('image_height')

    bounding_box_str = request.form.get('largest_box')
    # Deserialize it back into a Python list or tuple
    largest_box = json.loads(bounding_box_str)

    # Get form data (POST request)
    drone_id = request.form['drone']
    altitude = request.form['altitude']

    # Get the selected drone's details from the database
    drone = Drone.query.get(drone_id)

    # Calculate actual dimensions (use your formula here)
    actual_width, actual_height, actual_perimeter, actual_area, gsd_l, gsd_w = real_dimensions(sensor_length=drone.sensor_length,
            sensor_width=drone.sensor_width,
            focal_length=drone.focal_length,
            altitude=altitude,
            image_width=image_width,
            image_height=image_height,
            pixel_width=pixel_width,
            pixel_height=pixel_height)
    
    session['actual_width'] = actual_width
    session['actual_height'] = actual_height
    session['actual_perimeter'] = actual_perimeter
    session['actual_area'] = actual_area
    session['gsd_l'] = gsd_l
    session['gsd_w'] = gsd_w

    return render_template('check_regulationss.html',processed_image=processed_image, actual_height=actual_height, actual_width=actual_width, 
                           actual_area=actual_area, actual_perimeter=actual_perimeter, pixel_height=pixel_height, pixel_width=pixel_width,
                           image_width=image_width, image_height=image_height, largest_box=largest_box,
                           find_actual_dim=True, actual_dimensions=True, further_analysis=True)


@app.route('/cont_further_analysis', methods=['POST'])
def cont_further_analysis():

    processed_image = session.get('processed_image')
    pixel_width = session.get('pixel_width')
    pixel_height = session.get('pixel_height')
    image_width = session.get('image_width')
    image_height = session.get('image_height')
    actual_width = session.get('actual_width')
    actual_height = session.get('actual_height')
    actual_area = session.get('actual_area')
    actual_perimeter = session.get('actual_perimeter')
    largest_box = session.get('largest_box')

    return render_template('check_regulationss.html',
                           processed_image=processed_image, actual_height=actual_height, actual_width=actual_width, 
                           actual_area=actual_area, actual_perimeter=actual_perimeter, pixel_height=pixel_height, pixel_width=pixel_width,
                           image_width=image_width, image_height=image_height, largest_box=largest_box,
                           find_actual_dim=True, actual_dimensions=True, further_analysis=True, compound_flag=True, road_width_flag=True)


@app.route('/process_compound_points', methods=['POST'])
def process_compound_points():
    # Get the points for each side from the form data
    side_1 = request.form.get('side_1')
    side_2 = request.form.get('side_2')
    side_3 = request.form.get('side_3')
    side_4 = request.form.get('side_4')

    # Process the coordinates (convert from string "x,y" to tuples)
    points = {
        'side_1': tuple(map(int, side_1.split(','))),
        'side_2': tuple(map(int, side_2.split(','))),
        'side_3': tuple(map(int, side_3.split(','))),
        'side_4': tuple(map(int, side_4.split(','))),
    }

    processed_image = session.get('processed_image')
    pixel_width = session.get('pixel_width')
    pixel_height = session.get('pixel_height')
    image_width = session.get('image_width')
    image_height = session.get('image_height')
    actual_width = session.get('actual_width')
    actual_height = session.get('actual_height')
    actual_area = session.get('actual_area')
    actual_perimeter = session.get('actual_perimeter')
    largest_box = session.get('largest_box')
    gsd_l = session.get('gsd_l')
    gsd_w = session.get('gsd_w')

    # Perform your processing with the points here
    min_distances = {}
    for side, point in points.items():
        min_distance = point_to_bounding_box_distance(side, point, largest_box, gsd_l, gsd_w)
        min_distances[side] = min_distance

    session['min_distances'] = min_distances

    # Return to some template, passing the processed points
    return render_template('check_regulationss.html', points=points, min_distances=min_distances,
                           processed_image=processed_image, actual_height=actual_height, actual_width=actual_width, 
                           actual_area=actual_area, actual_perimeter=actual_perimeter, pixel_height=pixel_height, pixel_width=pixel_width,
                           image_width=image_width, image_height=image_height, largest_box=largest_box,
                           find_actual_dim=True, actual_dimensions=True, further_analysis=True, min_compound_flag=True, road_width_flag=True)


@app.route('/calculate_road_width', methods=['POST'])
def calculate_road_width():
    # Get the two road points from the form
    road_point_1 = request.form.get('road_point_1')
    road_point_2 = request.form.get('road_point_2')

    # Get the selected axis (x-axis or y-axis)
    axis = request.form.get('axis')

    # Convert the points from string format "x,y" to tuples
    point1 = tuple(map(int, road_point_1.split(',')))
    point2 = tuple(map(int, road_point_2.split(',')))

    processed_image = session.get('processed_image')
    pixel_width = session.get('pixel_width')
    pixel_height = session.get('pixel_height')
    image_width = session.get('image_width')
    image_height = session.get('image_height')
    actual_width = session.get('actual_width')
    actual_height = session.get('actual_height')
    actual_area = session.get('actual_area')
    actual_perimeter = session.get('actual_perimeter')
    largest_box = session.get('largest_box')
    gsd_l = session.get('gsd_l')
    gsd_w = session.get('gsd_w')

    # Calculate the distance (road width) between the two points
    actual_road_width = calculate_distance(point1, point2, gsd_l, gsd_w, axis)

    # Pass the road width back to the template for display
    return render_template('check_regulationss.html', actual_road_width=actual_road_width, axis=axis,
                           processed_image=processed_image, actual_height=actual_height, actual_width=actual_width, 
                           actual_area=actual_area, actual_perimeter=actual_perimeter, pixel_height=pixel_height, pixel_width=pixel_width,
                           image_width=image_width, image_height=image_height, largest_box=largest_box,
                           find_actual_dim=True, actual_dimensions=True, further_analysis=True, min_compound_flag=True, road_width_flag=True)





@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



@app.route('/upload-authorization-image', methods=['POST'])
def upload_authorization_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform building detection
        predicted_mask_image_filename, bounding_boxes = detect_buildings(file_path, model)

        return render_template('check_authorization.html', predicted_mask_image_filename=predicted_mask_image_filename, bounding_boxes=bounding_boxes, file_path=file_path)

    return redirect(request.url)









def detect_buildings(image_path, model):
    # Load your TensorFlow model
    #model = tf.keras.models.load_model('models/building_detection.h5')  # model path

    # Read and preprocess the image using OpenCV
    image = cv2.imread(image_path)

    # Dimensions of the original tif image
    original_width = image.shape[1]
    original_height = image.shape[0]

    input_image = preprocess_image(image_path)
    
    predicted_mask = model.predict(input_image)
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binary mask thresholding

    # Assuming predicted_mask_single is the 2D predicted mask
    predicted_mask_single = predicted_mask[0, :, :, 0]  # Extract the first mask for processing

    # Convert the mask to a format OpenCV can save and display
    # Ensure the mask has pixel values in the range [0, 255]
    predicted_mask_image = predicted_mask_single * 255  # Scale the mask values to 0-255

    # Check if the mask is single-channel (grayscale) and save it
    if len(predicted_mask_image.shape) == 2:  # Grayscale mask
        predicted_mask_image = cv2.cvtColor(predicted_mask_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR for display


    # Save the predicted mask as image (in tif format)
    predicted_image_filename = 'predicted_' + os.path.basename(image_path)
    predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_image_filename)
    save_processed_image_with_geotags(image_path,predicted_mask_image,predicted_image_path)
    #cv2.imwrite(predicted_image_path, predicted_mask_image)

    # Convert predicted mask image to .png or .jpg
    jpg_filename = predicted_image_filename.rsplit('.', 1)[0] + '.jpg'
    jpg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], jpg_filename)
    cv2.imwrite(jpg_filepath, predicted_mask_image)

    bounding_boxes = detect_bounding_boxes(predicted_mask_image, jpg_filepath, original_width, original_height)

    return jpg_filename, bounding_boxes




@app.route('/get_coordinates_from_pixel', methods=['POST'])
def get_coordinates_from_pixel():
    # Get the pixel coordinates from the form
    bounding_boxes = request.form.get('bounding_boxes')
    bounding_boxes = json.loads(bounding_boxes)
    image_path = request.form.get('file_path')  # Path to the tif image
    predicted_mask_image_filename = request.form.get('predicted_mask_image_filename')

    # Convert pixel coordinates to geographical coordinates
    lat_long_pairs = pixel_to_coordinates(bounding_boxes, image_path)
    addresses = coordinates_to_address(lat_long_pairs)
    
    # Zip the lat_long_pairs and addresses in the route itself
    coordinates_and_addresses = list(zip(lat_long_pairs, addresses))

    # Return the result back to the template (re-render the page with coordinates and address)
    return render_template('check_authorization.html', lat_long_pairs=lat_long_pairs, addresses=addresses,
                           predicted_mask_image_filename=predicted_mask_image_filename,  # Display the mask again
                           image_path=image_path,bounding_boxes=bounding_boxes, coordinates_and_addresses=coordinates_and_addresses,
                           enumerate=enumerate)



@app.route('/end-session', methods=['POST'])
def end_session():
    # Clear all session variables
    session.clear()

    # Redirect or render the home page 
    return redirect(url_for('check_regulations'))  



















if __name__ == '__main__':
    app.run(debug=True)

