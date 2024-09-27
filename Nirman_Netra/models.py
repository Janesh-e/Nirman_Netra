from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy object
db = SQLAlchemy()

# Define the Drone model
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
