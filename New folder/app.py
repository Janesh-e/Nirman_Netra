from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, jsonify, send_file
from sqlalchemy import case
from models import db, User, Drone, Sector, Survey, Building
from datetime import datetime
import json
import random
import sys
sys.path.append('C:\\Users\\ekamb\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages')
import os
import cv2
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np

from image_processing.image_processing import process_image, real_dimensions, detect_bounding_boxes, point_to_bounding_box_distance, calculate_distance, send_image_to_deepai
from image_processing.model_utils import preprocess_image, preprocess_image_cd, convert_to_grayscale, preprocess_image_wd, post_process_wmask, post_process_bmask, combine_and_visualize_masks, calculate_building_to_waterbody_distances, calculate_real_dimensions_from_bounding_box, draw_bounding_boxes
from image_processing.reverse_geocoding import save_processed_image_with_geotags, pixel_to_coordinates, coordinates_to_address, calculateHeight, resize_geotiff
from image_processing.report import create_pdf, generate_pdf_report

from sam_utils import initialize_sam, generate_masks_with_boxes, generate_masks_with_points, save_mask_as_image 

app = Flask(__name__)

# Set the secret key to some random bytes or any secret string
app.secret_key = '1234567890'  

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/drone_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# DeepAI API Key
DEEPAI_API_KEY = 'e6ee2a29-7dc9-4204-bc1c-ffe3c8500bb6'  
DEEPAI_API_URL = 'https://api.deepai.org/api/torch-srgan'  

# Initialize the database with the app
db.init_app(app)

# Create tables in the database
#with app.app_context():
 #   db.create_all()

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image formats
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'tiff', 'tif', 'png', 'PNG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the Change Detection Model
change_detection_model = tf.keras.models.load_model('models/unet_building_change_detection1.h5')

# Building Detection Model
model = tf.keras.models.load_model('models/RIO_building_detect_model1.keras')

# Water Body Detection Model
waterbody_model = tf.keras.models.load_model('models/water_model_heavy_rgb.h5')

# Initialize SAM model
sam_predictor = initialize_sam(model_type="vit_h", checkpoint_path="C:/Users/ekamb/Documents/segment-anything-main/segment-anything-main/weights/sam_vit_h_4b8939.pth", device="cuda")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find user by username
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not fullname or not username or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('register'))
        # Check if username or email already exists
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists!', 'danger')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(fullname=fullname, username=username, email=email)
        new_user.set_password(password)  # Save hashed password

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    surveys = db.session.query(Survey.id,Survey.name,Survey.date).group_by(Survey.name).order_by(Survey.date.desc()).all()
    sectors = db.session.query(Sector.name,Sector.id).all()
    selected_survey = None
    if request.method == 'GET':
        # Set the default survey as the most recent one (fetch its ID)
        recent_survey = Survey.query.order_by(Survey.date.desc()).first()
        selected_survey = recent_survey.id if recent_survey else None  # Ensure it's an ID
    if request.method == 'POST':
        selected_survey = int(request.form.get('survey'))  # Default to 0 if not provided

    return render_template('dashboard.html', selected_survey=selected_survey, surveys=surveys, sectors=sectors)




@app.route('/check-authorization', methods=['GET','POST'])
def check_authorization():
    if request.method == 'GET':
        # Fetch all surveys and sectors
        surveys = db.session.query(Survey.id,Survey.name,Survey.date).group_by(Survey.name).order_by(Survey.date.desc()).all()
        sectors = Sector.query.all()
        return render_template('check_authorization.html', surveys=surveys, sectors=sectors, len=len)
    if request.method == 'POST':
        # Handle form submission
        survey_name = request.form['survey']
        sector_id = request.form['sector']

        # Fetch the survey
        survey = Survey.query.filter_by(name=survey_name, sector_id=int(sector_id)).first()

        if not survey:
            flash("Invalid survey or sector selected.", "error")
            return redirect('/check_authorization')
        
        # Add to session
        session['survey_id'] = survey.id
        session['sector_id'] = sector_id
        
        # Perform authorization check using the orthophoto
        orthophoto_path = survey.orthophoto_path
        # Read the original orthophoto to get its size
        original_orthophoto = cv2.imread(orthophoto_path)
        filename = os.path.basename(orthophoto_path)
        original_height, original_width, _ = original_orthophoto.shape

        '''
        deepai_processed_image = send_image_to_deepai(orthophoto_path)
        # Resize the DeepAI processed image to the size of the original orthophoto
        resized_deepai_processed_image = cv2.resize(deepai_processed_image, (original_width, original_height))

        if deepai_processed_image is None:
            flash("Failed to process image using DeepAI API.", "error")
            return redirect('/check_authorization')

        # Save the processed image
        deepai_processed_image_filename = f"deepai_processed_{os.path.basename(orthophoto_path)}"
        deepai_processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], deepai_processed_image_filename)
        #cv2.imwrite(deepai_processed_image_path, deepai_processed_image)
        save_processed_image_with_geotags(orthophoto_path, resized_deepai_processed_image, deepai_processed_image_path)'
        '''
        # Generate the output path by appending "_resized" to the original filename
        #tiff_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_resized.tif" )
        #resize_geotiff(orthophoto_path,tiff_output_path, 438, 406)
        #orthophoto_path = tiff_output_path

        # Perform building detection
        predicted_mask_image_filename, bb_boxes = detect_buildings(orthophoto_path, model)

        # Perform processing for water body
        image = cv2.imread(orthophoto_path, cv2.IMREAD_UNCHANGED)
        png_filename = os.path.splitext(filename)[0]
        png_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{png_filename}.png")
        cv2.imwrite(png_path, image)
        png_filename = os.path.basename(png_path)

        grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], f"grayscale_{filename}")
        convert_to_grayscale(orthophoto_path, save_path=grayscale_path)
        preprocessed_image_wd = preprocess_image_wd(grayscale_path)
        preprocessed_image_bd = preprocess_image(grayscale_path)

        # Predict masks
        waterbody_pred = waterbody_model.predict(preprocessed_image_wd)
        building_pred = model.predict(preprocessed_image_bd)


        response_data = {'distances': []}
        coordinates_and_addresses_with_dimensions = []
        lat_long_pairs = []
        cx_cy_pairs = []
        addresses = []
        building_dimensions = []
        results = []
        combined_mask_filename = ""
        distance_visualisation_filename = ""
        water_distance_list = []
        flag=0

        wmask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'wmask_{filename}')
        waterbody_mask = post_process_wmask(waterbody_pred, wmask_path)
        #print(np.count_nonzero(waterbody_mask))
        if np.count_nonzero(waterbody_mask) == 0:
            survey.is_water_body_detected = False
            # Calculate Real Dimensions
            building_dimensions = calculate_real_dimensions_from_bounding_box(bb_boxes, orthophoto_path)

            lat_long_pairs,cx_cy_pairs = pixel_to_coordinates(bb_boxes, orthophoto_path)
            addresses = coordinates_to_address(lat_long_pairs)

            for building, (lat_long, address) in zip(building_dimensions, zip(lat_long_pairs, addresses)):
                dimensions = {
                    "real_width": building["real_width"],  
                    "real_height": building["real_height"]  
                }
                coordinates_and_addresses_with_dimensions.append((lat_long, address, dimensions))
        else:
            flag=1
            survey.is_water_body_detected = True
            # Post-process masks
            wmask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'wmask_{filename}')
            bmask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'bmask_{filename}')
            waterbody_mask = post_process_wmask(waterbody_pred, wmask_path)
            building_mask, bb_boxes = post_process_bmask(building_pred, bmask_path)

            # Calculate Real Dimensions
            building_dimensions = calculate_real_dimensions_from_bounding_box(bb_boxes, orthophoto_path)

            lat_long_pairs,cx_cy_pairs = pixel_to_coordinates(bb_boxes, orthophoto_path)
            addresses = coordinates_to_address(lat_long_pairs)

            for building, (lat_long, address) in zip(building_dimensions, zip(lat_long_pairs, addresses)):
                dimensions = {
                    "real_width": building["real_width"],  # Access dictionary key
                    "real_height": building["real_height"]  # Access dictionary key
                }
                coordinates_and_addresses_with_dimensions.append((lat_long, address, dimensions))


            # Combine and visualize masks
            combined_mask = combine_and_visualize_masks(
                waterbody_mask, building_mask, output_path=os.path.join(app.config['UPLOAD_FOLDER'], f'combined_mask_{filename}')
            )
            distance_visualisation_path = os.path.join(app.config['UPLOAD_FOLDER'], f'distance_vis_{filename}')
            # Calculate distances from building centroids to waterbody boundaries
            results = calculate_building_to_waterbody_distances(building_mask, waterbody_mask, orthophoto_path,distance_visualisation_path)
            combined_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'combined_mask_{filename}')
            # Prepare response
            response_data = {
                'distances': [
                    {'address': centroid, 'distance': distance} for centroid, distance in results
                ]
            }
            for centroid, distance in results:
                water_distance_list.append(distance)
            distance_visualisation_filename = os.path.basename(distance_visualisation_path)
            combined_mask_filename = os.path.basename(combined_mask_path)
            distance_visualisation_filename = distance_visualisation_filename.rsplit('.', 1)[0] + '.jpg'
            combined_mask_filename = combined_mask_filename.rsplit('.', 1)[0] + '.jpg'
        
        if(flag==0):
            for i in range(len(bb_boxes)):
                water_distance_list.append(-1)

        ctr = 0
        
        buildings_data = []
        # Save to Database
        for i, (bb_box, dimensions, (lat_long, address)) in enumerate(zip(bb_boxes, building_dimensions, zip(lat_long_pairs, addresses))):
            is_authorized_status = random.choice([0, 1]) 
            building = Building(
                building_id=i+1,
                survey_id=survey.id,  
                sector_id=sector_id,  
                image_path="-",
                width=dimensions["real_width"],  
                length=dimensions["real_height"],  
                latitude=lat_long[0],  
                longitude=lat_long[1],  
                central_x = cx_cy_pairs[i][0],
                central_y = cx_cy_pairs[i][1],
                address=address,
                is_authorized=is_authorized_status, 
                distance_from_waterbody = water_distance_list[ctr],
                detection_date=datetime.utcnow()
            )
            db.session.add(building)
            # Store data to send to the frontend
            buildings_data.append({
            "id": building.building_id,
            "latitude": lat_long[0],
            "longitude": lat_long[1],
            "width": dimensions["real_width"],
            "height": dimensions["real_height"],
            "address": address,
            "is_authorized": is_authorized_status  # Include authorization status
            })
            ctr=ctr+1

        # Update detection_date and total_buildings_detected
        survey.detection_date = datetime.utcnow()
        survey.total_buildings_detected = len(bb_boxes)
        survey.status = "COMPLETED"

        db.session.commit()

        # SAM 
        # Convert [x, y, width, height] to [x1, y1, x2, y2]
        bb_boxes_xyxy = [[x-10, y-10, x + width+10, y + height+10] for x, y, width, height in bb_boxes]

        # Generate SAM masks using bounding boxes
        sam_masks_with_boxes = generate_masks_with_boxes(sam_predictor, orthophoto_path, bb_boxes_xyxy)
        for i, mask in enumerate(sam_masks_with_boxes):
            save_mask_as_image(mask, os.path.join(app.config['UPLOAD_FOLDER'], f'sam_mask_box_{i}_{filename}.png'))
        print("SAM masks with boxes saved successfully")
        sam_filename = f'sam_mask_box_0_{filename}.png'
        # Generate SAM masks using single points (centroids)
        #centroids = [(bb[0] + bb[2] // 2, bb[1] + bb[3] // 2) for bb in bb_boxes_xyxy]  # Calculate centroids
        #sam_masks_with_points = generate_masks_with_points(sam_predictor, orthophoto_path, centroids)
        #for i, mask in enumerate(sam_masks_with_points):
            #save_mask_as_image(mask, os.path.join(app.config['UPLOAD_FOLDER'], f'sam_mask_point_{i}.png'))
        #print("SAM masks with centroids saved successfully")

        # Render the result on the page
        return render_template('check_authorization.html',result=response_data, combined_mask_filename=combined_mask_filename,distance_visualisation_filename=distance_visualisation_filename,
                                   predicted_mask_image_filename=predicted_mask_image_filename, bounding_boxes=bb_boxes, building_dimensions=building_dimensions, file_path=orthophoto_path,coordinates_and_addresses_with_dimensions=coordinates_and_addresses_with_dimensions, enumerate=enumerate, len=len,
                                   sam_filename=sam_filename, original_filename=png_filename,buildings_data=buildings_data)



@app.route('/generate-report/<int:survey_id>/<int:sector_id>', methods=['GET'])
def generate_report(survey_id, sector_id):
    try:
        # Define the path for the PDF
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'report_survey_{survey_id}_sector_{sector_id}.pdf')

        # Generate the PDF report
        generate_pdf_report(survey_id, sector_id, output_path)

        # Serve the PDF file
        return send_file(output_path, as_attachment=True, download_name=f'report_survey_{survey_id}_sector_{sector_id}.pdf')
    except Exception as e:
        flash(f"Error generating report: {str(e)}", "error")
        return redirect('/check_authorization')








@app.route('/get_coordinates_from_pixel', methods=['POST'])
def get_coordinates_from_pixel():
    # Get the pixel coordinates from the form
    bounding_boxes = request.form.get('bounding_boxes')
    bounding_boxes = json.loads(bounding_boxes)
    image_path = request.form.get('file_path')  # Path to the tif image
    predicted_mask_image_filename = request.form.get('predicted_mask_image_filename')

    # Convert pixel coordinates to geographical coordinates
    lat_long_pairs,cx_cy_pairs = pixel_to_coordinates(bounding_boxes, image_path)
    addresses = coordinates_to_address(lat_long_pairs)
    
    # Zip the lat_long_pairs and addresses in the route itself
    coordinates_and_addresses = list(zip(lat_long_pairs, addresses))

    # Insert detected buildings into the database
    for i, (lat_long, address) in enumerate(coordinates_and_addresses):
        new_building = Building(
            building_id=i+1,
            sector_id=session.get('sector_id'),
            survey_id=session.get('survey_id'),
            image_path = '-',
            bounding_box_path = '-',
            latitude=lat_long[0],
            longitude=lat_long[1],
            address=address,
            is_authorized=False,
            regulation_violations=False,
            central_x = cx_cy_pairs[i][0],
            central_y = cx_cy_pairs[i][1]
        )
        db.session.add(new_building)

    db.session.commit()  # Save all buildings to the database


    # Return the result back to the template (re-render the page with coordinates and address)
    return render_template('check_authorization.html', lat_long_pairs=lat_long_pairs, addresses=addresses,
                           predicted_mask_image_filename=predicted_mask_image_filename,  # Display the mask again
                           image_path=image_path,bounding_boxes=bounding_boxes, coordinates_and_addresses=coordinates_and_addresses,
                           enumerate=enumerate)



@app.route('/change_detection', methods=['GET', 'POST'])
def change_detection():
    if request.method == 'GET':
        # Fetch all unique surveys and sectors for dropdown options
        surveys = db.session.query(Survey.id,Survey.name,Survey.date).group_by(Survey.name).order_by(Survey.date.desc()).all()
        sectors = db.session.query(Sector.name).all()
        return render_template('change_detection.html', surveys=surveys, sectors=sectors)

    if request.method == 'POST':
        # Retrieve form data for before and after surveys and sectors
        before_survey_name = request.form.get('before_survey')
        before_sector_name = request.form.get('before_sector')
        after_survey_name = request.form.get('after_survey')
        after_sector_name = request.form.get('after_sector')

        # Fetch the corresponding orthophoto paths from the Survey table
        before_survey = (
            db.session.query(Survey)
            .join(Sector, Survey.sector_id == Sector.id)
            .filter(Survey.name == before_survey_name, Sector.name == before_sector_name)
            .first()
        )
        after_survey = (
            db.session.query(Survey)
            .join(Sector, Survey.sector_id == Sector.id)
            .filter(Survey.name == after_survey_name, Sector.name == after_sector_name)
            .first()
        )
        # Check if both surveys and their orthophotos are found
        if not before_survey or not after_survey:
            return render_template(
                'change_detection.html',
                surveys=db.session.query(Survey.name).distinct().all(),
                sectors=db.session.query(Sector.name).all(),
                error="One or both orthophotos could not be found.",
            )
        
        before_image_path = before_survey.orthophoto_path
        after_image_path = after_survey.orthophoto_path

        # Generate a filename based on before and after images
        before_filename = os.path.splitext(os.path.basename(before_image_path))[0]
        after_filename = os.path.splitext(os.path.basename(after_image_path))[0]

        # Preprocess the images
        img1 = preprocess_image_cd(before_image_path)
        img2 = preprocess_image_cd(after_image_path)

        # Predict changes
        input_data = np.concatenate([img1, img2], axis=-1)  # Concatenate along channel dimension
        prediction_mask = change_detection_model.predict(input_data)[0]

        # Post-process prediction to create mask image
        prediction_mask = (prediction_mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0-255

        change_mask_filename = f"{before_filename}_{after_filename}_change_mask.png"
        change_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], change_mask_filename)
        cv2.imwrite(change_mask_path, prediction_mask)

        # Calculate the percentage of change
        total_pixels = prediction_mask.size
        changed_pixels = np.count_nonzero(prediction_mask)  # Count white pixels
        change_percentage = (changed_pixels / total_pixels) * 100
        print(before_filename)
        return render_template(
            'change_detection.html',
            surveys=db.session.query(Survey.name).distinct().all(),
            sectors=db.session.query(Sector.name).all(),
            change_mask_path=change_mask_path,
            change_percentage=change_percentage,change_mask_filename=change_mask_filename,
            before_filename=f"{before_filename}.png", after_filename=f"{after_filename}.png"
        )


@app.route('/waterbody_detect', methods=['POST','GET'])
def waterbody_detect():
    if request.method == 'GET':
        # Fetch all surveys and sectors
        surveys = db.session.query(Survey.id,Survey.name,Survey.date).group_by(Survey.name).order_by(Survey.date.desc()).all()
        sectors = Sector.query.all()
        return render_template('waterbodydetect.html', surveys=surveys, sectors=sectors)
    
    if request.method == 'POST':
        # Handle form submission
        survey_name = request.form['survey']
        sector_id = request.form['sector']

        # Fetch the survey
        survey = Survey.query.filter_by(name=survey_name, sector_id=int(sector_id)).first()

        if not survey:
            flash("Invalid survey or sector selected.", "error")
            return redirect('/waterbody_detect')
        
        # Perform authorization check using the orthophoto
        orthophoto_path = survey.orthophoto_path

        # Perform processing
        try:
            # Preprocess the image
            filename = os.path.basename(orthophoto_path)
            grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], f"grayscale_{filename}")
            convert_to_grayscale(orthophoto_path, save_path=grayscale_path)
            preprocessed_image_wd = preprocess_image_wd(grayscale_path)
            preprocessed_image_bd = preprocess_image(grayscale_path)

            # Predict masks
            waterbody_pred = waterbody_model.predict(preprocessed_image_wd)
            building_pred = model.predict(preprocessed_image_bd)
            # Post-process masks
            wmask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'wmask_{filename}')
            bmask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'bmask_{filename}')
            waterbody_mask = post_process_wmask(waterbody_pred, wmask_path)
            building_mask = post_process_bmask(building_pred, bmask_path)

            # Combine and visualize masks
            combined_mask = combine_and_visualize_masks(
                waterbody_mask, building_mask, output_path=os.path.join(app.config['UPLOAD_FOLDER'], f'combined_mask_{filename}')
            )
            distance_visualisation_path = os.path.join(app.config['UPLOAD_FOLDER'], f'distance_vis_{filename}')
            # Calculate distances from building centroids to waterbody boundaries
            results = calculate_building_to_waterbody_distances(building_mask, waterbody_mask, orthophoto_path,distance_visualisation_path)
            combined_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'combined_mask_{filename}')
            # Prepare response
            response_data = {
                'distances': [
                    {'address': centroid, 'distance': distance} for centroid, distance in results
                ]
            }
            distance_visualisation_filename = os.path.basename(distance_visualisation_path)
            combined_mask_filename = os.path.basename(combined_mask_path)
            distance_visualisation_filename = distance_visualisation_filename.rsplit('.', 1)[0] + '.jpg'
            combined_mask_filename = combined_mask_filename.rsplit('.', 1)[0] + '.jpg'

            # Render the result on the page
            return render_template('waterbodydetect.html',result=response_data, combined_mask_filename=combined_mask_filename,distance_visualisation_filename=distance_visualisation_filename)

        except Exception as e:
            return render_template('waterbodydetect.html', result={'error': str(e)})




















@app.route('/check-regulations-choice', methods=['GET', 'POST'])
def check_regulations_choice():
    if request.method == 'POST':
        # Check which button was clicked
        if 'automated_mode' in request.form:
            return redirect(url_for('check_regulations'))  # Redirect to automated mode
        elif 'manual_mode' in request.form:
            return redirect(url_for('manual_mode'))  # Redirect to manual mode

    # Render choice form if GET request
    return render_template('check_regulations_choice.html')



@app.route('/check-regulations', methods=['GET', 'POST'])
def check_regulations():
    if request.method == 'GET':
        # Fetch all surveys and sectors
        surveys = db.session.query(Survey.id,Survey.name,Survey.date).group_by(Survey.name).order_by(Survey.date.desc()).all()
        sectors = Sector.query.all()
        return render_template('check_regulationss.html', surveys=surveys, sectors=sectors)
    if request.method == 'POST':
        # Handle form submission
        survey_name = request.form['survey']
        sector_id = request.form['sector']
        # Fetch the survey
        survey = Survey.query.filter_by(name=survey_name, sector_id=int(sector_id)).first()
        session['building_survey_id'] = survey.id
        session['building_sector_id'] = sector_id

        # Handle form submission for uploading the image, calculating dimensions, etc.
        return redirect(url_for('calculate_actual_dimensions_check'))
    else:
        # Render the initial check regulations page
        return render_template('check_regulationss.html')
    
@app.route('/manual-mode')
def manual_mode():
    # Render the manual mode page
    return render_template('manual_mode.html')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    survey_name = request.form['survey']
    sector_id = request.form['sector']
    survey = Survey.query.filter_by(name=survey_name, sector_id=int(sector_id)).first()
    session['building_survey_id'] = survey.id
    session['building_sector_id'] = sector_id
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
    building_survey_id = session.get('building_survey_id')
    building_sector_id = session.get('building_sector_id')

    # Fetch the survey
    survey = Survey.query.filter_by(id=int(building_survey_id), sector_id=int(building_sector_id)).first()

    orthophoto_path = survey.orthophoto_path

    height_floors = calculateHeight(orthophoto_path,27.711805795237414, 85.30023082960888)

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
                           find_actual_dim=True, actual_dimensions=True, further_analysis=True, height_floors=height_floors)


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







@app.route('/end-session', methods=['POST'])
def end_session():
    session.clear()
    return redirect(url_for('check_regulations'))  


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


@app.route('/data_upload', methods=['GET'])
def data_upload():
    tables = ['Survey', 'Sector', 'Building', 'Drone']  # Add table names as strings
    selected_table = None
    # Fetch sectors to populate the dropdown for surveys
    sectors = Sector.query.all()
    return render_template('data_upload.html', sectors=sectors, tables=tables, selected_table=selected_table)

@app.route('/add_sector', methods=['POST'])
def add_sector():
    sector_name = request.form['sector_name']
    sector_description = request.form.get('sector_description')

    # Create and save the new sector
    new_sector = Sector(name=sector_name, description=sector_description)
    db.session.add(new_sector)
    db.session.commit()
    flash(f"Sector '{sector_name}' added successfully.", "success")
    return redirect('/data_upload')

@app.route('/add_drone', methods=['POST'])
def add_drone():
    drone_name = request.form['drone_name']
    sensor_length = request.form['sensor_length']
    sensor_width = request.form['sensor_width']
    focal_length = request.form['focal_length']

    # Create and save the new drone
    new_drone = Drone(name=drone_name, sensor_length=sensor_length, sensor_width=sensor_width, focal_length=focal_length)
    db.session.add(new_drone)
    db.session.commit()
    flash(f"Drone '{drone_name}' added successfully.", "success")
    return redirect('/data_upload')

@app.route('/add_survey', methods=['POST'])
def add_survey():
    survey_name = request.form['survey_name']
    survey_date = request.form['survey_date']
    sector_id = request.form['survey_sector']
    orthophoto = request.files['orthophoto']

    try:
        # Convert the date string to a datetime object
        survey_date_obj = datetime.strptime(survey_date, '%Y-%m-%d')
        if orthophoto and allowed_file(orthophoto.filename):
            filename = secure_filename(orthophoto.filename)
            orthophoto_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            orthophoto.save(orthophoto_path)

            # Create and save the new survey
            new_survey = Survey(name=survey_name, date=survey_date_obj, sector_id=sector_id, orthophoto_path=orthophoto_path)
            db.session.add(new_survey)
            db.session.commit()
            flash(f"Survey '{survey_name}' added successfully.", "success")
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        return redirect('/data_upload')
    
    return redirect('/data_upload')


@app.route('/view_table', methods=['GET', 'POST'])
def view_table():
    tables = ['Survey', 'Sector', 'Building', 'Drone']  # Add table names as strings
    selected_table = None
    records = []
    page = 1  # Default to page 1
    per_page = 10  # Records per page

    if request.method == 'POST':
        selected_table = request.form.get('table')  # Get the selected table name
        page = int(request.form.get('page', 1))  # Get the current page number

        # Dynamically fetch records based on the selected table
        if selected_table == 'Survey':
            query = Survey.query
        elif selected_table == 'Sector':
            query = Sector.query
        elif selected_table == 'Building':
            query = Building.query
        elif selected_table == 'Drone':
            query = Drone.query
        else:
            flash("Invalid table selected.", "error")
            return redirect('/data_upload')

        # Pagination logic
        total_records = query.count()
        records = query.offset((page - 1) * per_page).limit(per_page).all()

        return render_template(
            'data_upload.html',
            tables=tables,
            selected_table=selected_table,
            records=records,
            total_records=total_records,
            per_page=per_page,
            current_page=page,
        )

    # Initial load
    return render_template('data_uplaod.html', tables=tables, selected_table=selected_table)


@app.template_filter('get_attr')
def get_attr(obj, attr_name):
    return getattr(obj, attr_name, None)






@app.route('/get_survey_completion_stats/<int:survey_id>')
def get_survey_completion_stats(survey_id):
    # Fetch all survey rows for the given survey ID
    surveys = Survey.query.filter_by(name=Survey.query.get(survey_id).name).all()
    if not surveys:
        return jsonify({'error': 'Survey not found'}), 404

    # Calculate statistics
    total_sectors = len(surveys)
    completed_sectors = sum(1 for survey in surveys if survey.status == 'COMPLETED')
    pending_sectors = total_sectors - completed_sectors

    data = {
        'survey_name': Survey.query.get(survey_id).name,
        'total_sectors': total_sectors,
        'completed_sectors': completed_sectors,
        'pending_sectors': pending_sectors,
    }
    return jsonify(data)


@app.route('/get_sector_building_stats/<int:survey_id>')
def get_sector_building_stats(survey_id):
    survey = Survey.query.get(survey_id)
    if not survey:
        return jsonify({'error': 'Survey not found'}), 404
    survey_name = survey.name

    # Fetch all survey IDs that share the same survey name
    surveys_with_same_name = Survey.query.filter_by(name=survey_name).all()
    survey_ids = [s.id for s in surveys_with_same_name]

    # Step 3: Fetch building data for all survey IDs, join with sectors, and group by sector
    stats = (
        db.session.query(
            Sector.name.label('sector_name'),
            db.func.sum(db.case((Building.is_authorized == True, 1), else_=0)).label('authorized_count'),
            db.func.sum(db.case((Building.is_authorized == False, 1), else_=0)).label('unauthorized_count')
        )
        .join(Building, Building.sector_id == Sector.id)
        .filter(Building.survey_id.in_(survey_ids))  # Filter by all survey IDs with the same name
        .group_by(Sector.name)  # Group by sector name
        .all()
    )
    # Step 4: Format data for the frontend
    formatted_data = {
        "sectors": [row.sector_name for row in stats],
        "authorized_counts": [row.authorized_count for row in stats],
        "unauthorized_counts": [row.unauthorized_count for row in stats],
    }
    print(formatted_data)
    return jsonify(formatted_data)



@app.route('/get_sector_violation_stats/<int:survey_id>')
def get_sector_violation_stats(survey_id):
    # Step 1: Fetch the survey name corresponding to the provided survey_id
    survey = Survey.query.get(survey_id)
    if not survey:
        return jsonify({'error': 'Survey not found'}), 404
    survey_name = survey.name

    # Step 2: Fetch all survey IDs that share the same survey name
    surveys_with_same_name = Survey.query.filter_by(name=survey_name).all()
    survey_ids = [s.id for s in surveys_with_same_name]

    # Step 3: Fetch building data for all survey IDs, join with sectors, and group by sector and violation status
    stats = (
        db.session.query(
            Sector.name.label('sector_name'),
            db.func.sum(db.case((Building.regulation_violations == True, 1), else_=0)).label('violation_count'),
            db.func.sum(db.case((Building.regulation_violations == False, 1), else_=0)).label('non_violation_count')
        )
        .join(Building, Building.sector_id == Sector.id)
        .filter(Building.survey_id.in_(survey_ids))  # Filter by all survey IDs with the same name
        .group_by(Sector.name)  # Group by sector name
        .all()
    )

    # Step 4: Format data for the frontend
    formatted_data = {
        "sectors": [row.sector_name for row in stats],
        "violation_counts": [row.violation_count for row in stats],
        "non_violation_counts": [row.non_violation_count for row in stats],
    }

    return jsonify(formatted_data)


@app.route('/get_sector_violation_check_stats/<int:survey_id>')
def get_sector_violation_check_stats(survey_id):
    # Step 1: Fetch the survey name corresponding to the provided survey_id
    survey = Survey.query.get(survey_id)
    if not survey:
        return jsonify({'error': 'Survey not found'}), 404

    survey_name = survey.name

    # Step 2: Fetch all survey IDs that share the same survey name
    surveys_with_same_name = Survey.query.filter_by(name=survey_name).all()
    survey_ids = [s.id for s in surveys_with_same_name]

    # Step 3: Fetch building data for all survey IDs, join with sectors, and group by sector and regulation check status
    stats = (
        db.session.query(
            Sector.name.label('sector_name'),
            db.func.sum(db.case((Building.regulation_violations.isnot(None), 1), else_=0)).label('checked_count'),
            db.func.sum(db.case((Building.regulation_violations.is_(None), 1), else_=0)).label('not_checked_count')
        )
        .join(Building, Building.sector_id == Sector.id)
        .filter(Building.survey_id.in_(survey_ids))  # Filter by all survey IDs with the same name
        .group_by(Sector.name)  # Group by sector name
        .all()
    )

    # Step 4: Format data for the frontend
    formatted_data = {
        "sectors": [row.sector_name for row in stats],
        "checked_counts": [row.checked_count for row in stats],
        "not_checked_counts": [row.not_checked_count for row in stats],
    }

    return jsonify(formatted_data)


@app.route('/get_building_trends')
def get_building_trends():
    # Query to calculate the number of unauthorized and violating buildings per survey
    trends_query = (
        db.session.query(
            Survey.name.label('survey_name'),
            db.func.sum(
                case(
                    (Building.is_authorized == False, 1),  # Unauthorized buildings
                    else_=0
                )
            ).label('unauthorized_count'),
            db.func.sum(
                case(
                    (Building.regulation_violations == True, 1),  # Violating buildings
                    else_=0
                )
            ).label('violating_count')
        )
        .join(Building, Survey.id == Building.survey_id)
        .group_by(Survey.name)
        .order_by(Survey.date)
    )

    trends_data = trends_query.all()

    # Format the data for the frontend
    data = {
        "surveys": [row.survey_name for row in trends_data],
        "unauthorized_counts": [row.unauthorized_count for row in trends_data],
        "violating_counts": [row.violating_count for row in trends_data],
    }

    return jsonify(data)





@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    # Fetch selected survey and sector IDs from the form
    survey_name = request.form.get('survey_name')
    sector_id = request.form.get('sector_id')

    # Validate the inputs
    if not survey_name or not sector_id:
        return jsonify({'error': 'Invalid survey or sector selection.'}), 400

    # Fetch the orthophoto path from the Survey table
    survey = Survey.query.filter_by(name=survey_name, sector_id=sector_id).first()
    if not survey or not survey.orthophoto_path:
        return jsonify({'error': 'Orthophoto not found for the selected survey and sector.'}), 404

    orthophoto_path = survey.orthophoto_path

    # Fetch unauthorized buildings from the Building table
    unauthorized_buildings = Building.query.filter_by(survey_id=survey.id, sector_id=sector_id, is_authorized=False).all()
    if not unauthorized_buildings:
        return jsonify({'error': 'No unauthorized buildings found for the selected survey and sector.'}), 404

    # Load the orthophoto
    orthophoto = cv2.imread(orthophoto_path)
    #orthophoto = cv2.resize(orthophoto, (256, 256))
    if orthophoto is None:
        return jsonify({'error': 'Failed to load orthophoto.'}), 500

    # Create a blank image for heatmap
    heatmap = np.zeros((orthophoto.shape[0], orthophoto.shape[1]), dtype=np.float32)

    halo_radius = 200  # This sets the radius of the influence around each central point
    max_intensity = 1  # Maximum intensity for the central pixel

    # Iterate through the buildings and mark their surrounding pixels in the raw heatmap
    for building in unauthorized_buildings:
        central_x = building.central_x
        central_y = building.central_y
    
        # Ensure the building has valid central pixel values
        if central_x is not None and central_y is not None:
            # Create a square region around the central point and set the values to 1
            for dx in range(-halo_radius, halo_radius + 1):
                for dy in range(-halo_radius, halo_radius + 1):
                    # Calculate the coordinates of the surrounding pixels
                    x = central_x + dx
                    y = central_y + dy
                
                    # Ensure the coordinates are within the bounds of the heatmap
                    if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                        # Calculate the distance from the center
                        distance = np.sqrt(dx**2 + dy**2)
                        # If within the halo radius, calculate intensity
                        if distance <= halo_radius:
                            intensity = max_intensity * (1 - distance / halo_radius)  # Linearly decrease intensity
                            heatmap[y, x] += intensity  # Accumulate intensity for overlapping regions


    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine the heatmap with the orthophoto
    heatmap_resized = cv2.resize(heatmap_colored, (orthophoto.shape[1], orthophoto.shape[0]))
    combined = cv2.addWeighted(orthophoto, 1, heatmap_resized, 0.5, 0)

    # Save the heatmap overlay image
    heatmap_filename = f"heatmap_survey{survey.id}_sector{sector_id}.png"
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_path, combined)

    # Return the path to display the heatmap in the dashboard
    return jsonify({'heatmap_path': heatmap_filename})




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

