from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

report_content = """
Day | Sleep Time | Phone Time | Washroom Visits | Posture While Lying | Walking Posture | Sleep Issues | Coughing | Sneezing | Facial Symptoms | Rashes | Patient Fainting | Patient Chest Pain | Conversation | Diet Timings | Avg. Blood Pressure (Systolic) | Avg. Blood Pressure (Diastolic) | Avg. Body Temperature | Heart Attack Risk | sugar levels
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Day 1 | 1.80 | 0.78 | 5 | Sideways | Normal | Restless | Mild | None | Sad | Mild | Yes | Yes | Slurred | Skipped | 111.00 | 78.00 | 98.42 | Low | 90 mg/dL
Day 2 | 1.75 | 0.70 | 3 | Straight | Slouched | None | Severe | Frequent | Pain | None | Yes | Yes | Normal | Irregular | 119.50 | 74.75 | 97.60 | Low | 160 mg\dL
Day 3 | 1.42 | 0.82 | 3 | Sideways | Slouched | Restless | None | Frequent | Neutral | Severe | Yes | Yes | Slurred | Irregular | 115.50 | 71.50 | 98.25 | High | 220 mg/dL
Day 4 | 0.98 | 0.68 | 4 | Sideways | Sideways | Interrupted | None | None | Pain | Severe | Yes | Yes | Slurred | Regular | 107.50 | 76.50 | 97.62 | Moderate | 90 mg/dL
Day 5 | 1.12 | 0.60 | 5 | Straight | Sideways | None | Mild | Mild | Neutral | None | Yes | Yes | Slurred | Regular | 122.75 | 74.50 | 98.65 | Moderate | 130 mg/dL
"""

def generate_pdf(report, output_filename="medical_report_a3.pdf"):
    # Use A3 landscape page for more width
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(A3), 
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CenteredTitle', parent=styles['Title'], alignment=TA_CENTER)
    title = Paragraph("Medical Report for John Doe", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))

    lines = report.strip().split('\n')
    data = [line.split(' | ') for line in lines if line.strip() and not line.startswith('---')]

    # Spread 19 columns evenly across wide A3 landscape width (~16.5 inch width minus margins)
    page_width = landscape(A3)[0] - inch  # Subtract margins (0.5 inch * 2)
    col_width = page_width / 19
    col_widths = [col_width] * 19

    wrapped_data = []
    for row in data:
        wrapped_row = [Paragraph(cell, styles['Normal']) for cell in row]
        wrapped_data.append(wrapped_row)

    table = Table(wrapped_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))

    elements.append(table)
    doc.build(elements)
    print(f"PDF generated: {output_filename}")

generate_pdf(report_content)
