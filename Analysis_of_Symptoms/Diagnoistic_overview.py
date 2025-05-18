from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

# Additional content
additional_content = """
*Medical Report for John Doe*
_________________________________________________

### Patient Information

* *Name*: John Doe
* *Age* : 58
* *Sex* : Male
* *Diagnosis: Stable (likely **Diabetes Mellitus* or *Hypotension* given symptoms)
* *Number of Medicines Given*: 0
* *Medicines Taken*: None

### Key Symptoms Used for Prediction

1. *Fainting* and *Chest Pain*: suggest cardiovascular or neurological issues.
2. *Severe Rashes*: could indicate allergic reactions or infections.
3. *Frequent Sneezing* and *Mild Coughing*: suggest respiratory issues.

### Structured Report

| Category                  | Observation          |
| *Glucose (Trips)*       | 2                    |
| *Sleep Time*            | 1.4 hours            |
| *Phone Time*            | 0.7 hours            |
| *Washroom Visits*       | 1                    |
| *Posture While Lying*   | Straight             |
| *Walking Posture*       | Slouched             |
| *Sleep Issues*          | Restless             |
| *Coughing*              | Mild                 |
| *Sneezing*              | Frequent             |
| *Facial Symptoms*       | Sad                  |
| *Rashes*                | Severe               |
| *Patient Fainting*      | Yes                  |
| *Patient Chest Pain*    | Yes                  |
| *Patient Conversation*  | Slurred              |
| *Diet Timings*          | Skipped              |

### Time Segments Observations

* *Morning to Afternoon*: Moderate
* *Afternoon to Evening*: Moderate
* *Evening to Night*: Moderate
* *Midnight to Morning*: Low

### Likely Disease Prediction

Based on the symptoms provided, the likely disease could be *Diabetes Mellitus* given the mention of *Glucose Trips, **nausea, and **dizziness. However, the combination of **chest pain, **fainting, and **severe rashes* could also suggest other conditions such as *Hypotension* or an *allergic reaction*. Further diagnosis is required for a definitive conclusion.
"""

# Function to parse structured report table from additional content
def parse_structured_report(content):
    lines = content.strip().split('\n')
    table_data = []
    in_table = False
    for line in lines:
        if line.startswith('| Category'):
            in_table = True
            headers = [h.strip() for h in line.split('|')[1:-1]]
            table_data.append(headers)
        elif in_table and line.startswith('|'):
            row = [r.strip() for r in line.split('|')[1:-1]]
            table_data.append(row)
        elif in_table and not line.startswith('|'):
            break
    return table_data

# Function to convert report to PDF
def generate_pdf(additional_content, output_filename="medical_report_patient.pdf"):
    doc = SimpleDocTemplate(output_filename, pagesize=landscape(A3), leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    heading_style = styles['Heading1']
    heading_style.alignment = TA_CENTER
    subheading_style = styles['Heading2']
    body_style = styles['Normal']
    list_style = styles['BodyText']
    list_style.leftIndent = 20

    # Add title
    elements.append(Paragraph("Medical Report for John Doe", heading_style))
    elements.append(Paragraph("_" * 199, body_style))
    elements.append(Spacer(1, 0.2*inch))

    # Patient Information
    elements.append(Paragraph("Patient Information", subheading_style))
    patient_info = [
        "<b>Name</b>: John Doe",
        "<b>Age</b> : 58",
        "<b>Sex</b> : Male",
        "<b>Diagnosis</b>: Stable (likely <b>Diabetes Mellitus</b> or <b>Hypotension</b> given symptoms)",
        "<b>Number of Medicines Given</b>: 0",
        "<b>Medicines Taken</b>: None"
    ]
    for info in patient_info:
        elements.append(Paragraph(f"• {info}", list_style))
    elements.append(Spacer(1, 0.2*inch))

    # Key Symptoms
    elements.append(Paragraph("Key Symptoms Used for Prediction", subheading_style))
    symptoms = [
        "<b>Fainting</b> and <b>Chest Pain</b>: suggest cardiovascular or neurological issues.",
        "<b>Severe Rashes</b>: could indicate allergic reactions or infections.",
        "<b>Frequent Sneezing</b> and <b>Mild Coughing</b>: suggest respiratory issues."
    ]
    for i, symptom in enumerate(symptoms, 1):
        elements.append(Paragraph(f"{i}. {symptom}", list_style))
    elements.append(Spacer(1, 0.2*inch))

    # Structured Report Table
    elements.append(Paragraph("Structured Report", subheading_style))
    structured_data = parse_structured_report(additional_content)
    structured_col_widths = [4*inch, 4*inch]  # Fixed widths for 2-column table
    structured_table_data = [[Paragraph(cell, body_style) for cell in row] for row in structured_data]
    structured_table = Table(structured_table_data, colWidths=structured_col_widths)
    structured_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # No WORDWRAP to ensure single-line text
    ]))
    elements.append(structured_table)
    elements.append(Spacer(1, 0.2*inch))

    # Time Segments Observations
    elements.append(Paragraph("Time Segments Observations", subheading_style))
    time_segments = [
        "<b>Morning to Afternoon</b>: Moderate",
        "<b>Afternoon to Evening</b>: Moderate",
        "<b>Evening to Night</b>: Moderate",
        "<b>Midnight to Morning</b>: Low"
    ]
    for segment in time_segments:
        elements.append(Paragraph(f"• {segment}", list_style))
    elements.append(Spacer(1, 0.2*inch))

    # Likely Disease Prediction
    elements.append(Paragraph("Likely Disease Prediction", subheading_style))
    prediction = (
        "Based on the symptoms provided, the likely disease could be <b>Diabetes Mellitus</b> given the mention of "
        "<b>Glucose Trips</b>, <b>nausea</b>, and <b>dizziness</b>. However, the combination of <b>chest pain</b>, "
        "<b>fainting</b>, and <b>severe rashes</b> could also suggest other conditions such as <b>Hypotension</b> or an "
        "<b>allergic reaction</b>. Further diagnosis is required for a definitive conclusion."
    )
    elements.append(Paragraph(prediction, body_style))
    elements.append(Spacer(1, 0.2*inch))

    # Build the PDF
    doc.build(elements)
    print(f"PDF generated: {output_filename}")

# Generate the PDF
generate_pdf(additional_content)