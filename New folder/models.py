from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize the SQLAlchemy object
db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    # Save hashed password
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Verify hashed password
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'




class Drone(db.Model):
    __tablename__ = 'drones'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    sensor_length = db.Column(db.Float, nullable=False)
    sensor_width = db.Column(db.Float, nullable=False)
    focal_length = db.Column(db.Float, nullable=False)

    def __init__(self, name, sensor_length, sensor_width, focal_length):
        self.name = name
        self.sensor_length = sensor_length
        self.sensor_width = sensor_width
        self.focal_length = focal_length

    def __repr__(self):
        return f'<Drone {self.name}>'



# Sector model to represent geographic regions/sectors
class Sector(db.Model):
    __tablename__ = 'sectors'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.String(300), nullable=True)

    # Relationships
    buildings = db.relationship('Building', backref='sector', lazy=True)



# Building Model
class Building(db.Model):
    __tablename__ = 'buildings'

    id = db.Column(db.Integer, primary_key=True)
    building_id = db.Column(db.Integer, nullable=False)
    sector_id = db.Column(db.Integer, db.ForeignKey('sectors.id'), nullable=False)
    survey_id = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    height = db.Column(db.Float, nullable=True)
    length = db.Column(db.Float, nullable=True)
    width = db.Column(db.Float, nullable=True)
    no_of_floors = db.Column(db.Integer, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    central_x = db.Column(db.Integer, nullable=True)
    central_y = db.Column(db.Integer, nullable=True)
    address = db.Column(db.String(255), nullable=False)
    is_authorized = db.Column(db.Boolean, nullable=False, default=False)
    regulation_violations = db.Column(db.Boolean, nullable=False, default=False)
    distance_from_waterbody = db.Column(db.Float, nullable=False)
    detection_date = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)


# Survey model to store metadata about each survey
class Survey(db.Model):
    __tablename__ = 'surveys'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    orthophoto_path = db.Column(db.String(200), nullable=False)
    sector_id = db.Column(db.Integer, db.ForeignKey('sectors.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='PENDING')
    change_percentage = db.Column(db.Float, nullable=True)
    detection_date = db.Column(db.DateTime, nullable=True, default=datetime.utcnow)
    total_buildings_detected = db.Column(db.Integer, nullable=True, default=0)
    is_water_body_detected = db.Column(db.Boolean, nullable=True, default=False)

    # Relationships
    buildings = db.relationship('Building', backref='survey', lazy=True)
    sector = db.relationship('Sector', backref='surveys')








