from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.lib.pagesizes import letter
from textwrap import wrap
from models import db, User, Drone, Sector, Survey, Building

def create_pdf(file_name, title, name, area, sector, survey_no, table_data, no_of_buildings, no_of_unauthorized, no_of_authorized, image_path):
    
    # Setup PDF
    pdf = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    # Add Title Bar
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, title)

    # Add Details Below Title Bar
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 100, f" City Name: {name}")
    pdf.drawString(50, height - 120, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    pdf.drawString(300, height - 120, f"Time: {datetime.now().strftime('%H:%M:%S')}")
    pdf.drawString(50, height - 140, f"Area: {area}")
    pdf.drawString(300, height - 140, f"Sector: {sector}")
    pdf.drawString(50, height - 160, f"Survey No: {survey_no}")

    # Add Image
    try:
        img_x, img_y = 50, height - 350  # Adjust coordinates for spacing
        img_width, img_height = 300, 200  # Adjust size as needed
        pdf.drawImage(image_path, img_x, img_y, img_width, img_height)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")

    # Add Table Header
    styles = getSampleStyleSheet()
    table_data.insert(0, ["Survey No", "Latitude", "Longitude", "Address"])
    table = Table(table_data, colWidths=[80, 100, 100, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#d3d3d3'),
        ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, '#000000')
    ]))

    # Draw Table on PDF
    table.wrapOn(pdf, width - 100, height - 500)
    table.drawOn(pdf, 50, height - 500)

    # Add Footer
    pdf.drawString(50, 100, f"No of Buildings: {no_of_buildings}")
    pdf.drawString(50, 80, f"No of Unauthorized: {no_of_unauthorized}")
    pdf.drawString(50, 60, f"No of Authorized: {no_of_authorized}")

    pdf.save()


def generate_pdf_report(survey_id, sector_id, output_path):
    # Fetch the survey, sector, and building details
    survey = Survey.query.get(survey_id)
    sector = Sector.query.get(sector_id)
    buildings = Building.query.filter_by(survey_id=survey_id, sector_id=sector_id).all()

    # Create a PDF canvas
    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Add the title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(30, height - 50, "Building Detection Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, height - 80, f"Survey: {survey.name}")
    pdf.drawString(30, height - 100, f"Sector: {sector.name}")
    pdf.drawString(30, height - 120, f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # Table header
    pdf.setFont("Helvetica-Bold", 10)
    y_position = height - 150
    pdf.drawString(30, y_position, "ID")
    pdf.drawString(80, y_position, "Address")
    pdf.drawString(400, y_position, "Lat, Long")
    pdf.drawString(550, y_position, "Status")

    # Draw a line below the header
    pdf.line(30, y_position - 10, 580, y_position - 10)
    y_position -= 30

    # Add building details
    pdf.setFont("Helvetica", 10)
    for building in buildings:
        if y_position < 50:  # Add a new page if space is insufficient
            pdf.showPage()
            y_position = height - 50

        pdf.drawString(30, y_position, str(building.building_id))
        # Wrap address text to fit within a width
        address_lines = wrap(building.address, width=50)  # Adjust `width` as per your page size
        for line in address_lines:
            pdf.drawString(80, y_position, line)
            y_position -= 15  # Adjust spacing between wrapped lines
        pdf.drawString(400, y_position + 15 * (len(address_lines) - 1), f"{building.latitude:.6f}, {building.longitude:.6f}")
        pdf.drawString(550, y_position, f"{building.is_authorized}")
        y_position -= 40

    # Save the PDF
    pdf.save()
