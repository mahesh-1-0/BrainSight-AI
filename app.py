from flask import Flask, render_template, request, session, send_file, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from datetime import datetime
import base64
import io
import re
import gc  # <--- CHANGED: Added garbage collection

# Load environment variables from .env file if available (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best_model_dense.h5"
LAST_CONV_LAYER = "top_activation"
CLASS_NAMES = ["brain_glioma", "brain_menin", "brain_tumor"]
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. AI analysis feature will not work.")
    print("Please set GEMINI_API_KEY environment variable or add it to .env file")
else:
    # Initialize Gemini
    genai.configure(api_key=GEMINI_API_KEY)

# Load model (global to avoid reloading)
model = None

if os.path.exists(MODEL_PATH):
    try:
        print("Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
else:
    print("Model file not found. Running in demo mode (no ML inference).")

# ===============================
# Brain Tumor Detection Functions
# ===============================
def load_image_from_file(file_path):
    """Load and preprocess image from file path"""
    original = cv2.imread(file_path)
    if original is None:
        raise ValueError("Image not found or invalid format")

    # Resize for display (High Res)
    original = cv2.resize(original, (512, 512))

    # Resize for Model Input (224, 224)
    img = cv2.resize(original, (224, 224))
    
    # Preprocessing (0-1 scale)
    img = img.astype(np.float32) / 255.0
    
    img = np.expand_dims(img, axis=0)
    return img, original

def gradcam(input_data, model, layer_name, class_idx):
    # <--- CHANGED: Updated inputs to accept dictionary or tensor
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    class_idx = int(class_idx)  # Ensure integer for indexing

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        
        # --- FIX FOR TENSORFLOW 2.16+ / KERAS 3 ---
        # Keras 3 may return predictions as a list. We extract the tensor.
        if isinstance(predictions, list):
            predictions = predictions[0]
        # ------------------------------------------

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    
    # Check if conv_outputs is a list (another Keras 3 quirk)
    if isinstance(conv_outputs, list):
        conv_outputs = conv_outputs[0]

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

    cam = np.maximum(cam.numpy(), 0)
    
    # Avoid division by zero
    if np.max(cam) == 0:
        return cam
        
    cam = cam / (np.max(cam) + 1e-8)

    return cam

def create_visual(original, heatmap, label):
    """Create visualization with heatmap overlay"""
    # Resize heatmap to 512x512 to match original
    heatmap = cv2.resize(heatmap, (512, 512))
    
    # Convert to uint8 (0-255)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply Color Map
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Soft blend
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    # Draw Yellow Circle
    _, mask = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        
        if r > 15:
            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 255), 3)

    # Add Label Text
    text = f"Prediction: {label}"
    cv2.rectangle(overlay, (10, 10), (360, 48), (0, 0, 0), -1)
    cv2.putText(
        overlay, text, (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2
    )

    return overlay

def process_brain_scan(image_path):
    """Process brain scan and return results"""
    
    if model is None:
        raise RuntimeError("ML model is not available in this deployment.")
    
    img_array, original = load_image_from_file(image_path)
    
    # <--- CHANGED: Fix for "Structure of inputs doesn't match" error
    # We check if the model expects named inputs and wrap the array in a dict
    input_data = img_array
    if hasattr(model, 'input_names') and model.input_names:
        # e.g. {'input_layer': img_array}
        input_data = {model.input_names[0]: img_array}

    # Predict using the properly formatted input
    preds = model.predict(input_data, verbose=0)

    # Handle Keras 3 returning a list
    if isinstance(preds, list):
        preds = preds[0]

    # Force scalar index (important for NumPy 2.x)
    class_idx = int(np.argmax(preds, axis=1)[0])

    confidence = float(preds[0][class_idx]) * 100
    label = CLASS_NAMES[class_idx]

    # <--- CHANGED: Added Memory Safety Block
    # If GradCAM crashes (due to memory), we catch it and continue
    try:
        heatmap = gradcam(input_data, model, LAST_CONV_LAYER, class_idx)
        final_img = create_visual(original, heatmap, label)
    except Exception as e:
        print(f"WARNING: GradCAM failed (likely low memory). Skipping heatmap. Error: {e}")
        # Fallback: Just return the original image with the label text drawn on it
        final_img = original.copy()
        cv2.rectangle(final_img, (10, 10), (360, 48), (0, 0, 0), -1)
        cv2.putText(
            final_img, f"Pred: {label} (No Heatmap)", (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2
        )
    
    # <--- CHANGED: Explicit memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()

    return final_img, label, confidence, preds[0]

# ===============================
# Gemini API Integration
# ===============================
def analyze_tumor_stage(image_path, prediction_label, confidence):
    """Use Gemini to analyze tumor stage from the image"""
    if not GEMINI_API_KEY:
        return "AI analysis unavailable: GEMINI_API_KEY not configured. Please set the environment variable."
    
    try:
        from PIL import Image as PILImage
        
        # Load the image for Gemini
        img = PILImage.open(image_path)
        
        prompt = f"""
        You are a medical AI assistant analyzing a brain MRI scan. 
        
        The AI model has detected: {prediction_label} with {confidence:.2f}% confidence.
        
        Based on the brain MRI scan image provided, please analyze:
        1. The stage/severity of the tumor (if present)
        2. The location and size of any abnormalities
        3. Clinical observations
        4. Recommendations for next steps
        
        Provide a professional medical analysis in a clear, structured format.
        Be concise but thorough. If no tumor is clearly visible, state that clearly.
       
        IMPORTANT FORMATTING INSTRUCTIONS:
        - Output the response in PLAIN TEXT only.
        - Do NOT use markdown formatting such as **bold** or ## headings.
        - Do NOT use asterisks (*) or hash signs (#) for formatting.
        - Use simple numbering (1., 2.) for lists.
        """
        
        # Try multiple model names in order of preference
        # Updated to use currently available models
        model_names = [
            'gemini-pro',           # Most stable and widely available
            'gemini-1.0-pro',       # Alternative stable option
            'gemini-2.5-flash-lite', # Newer, faster option
            'gemini-2.5-flash',     # Newer option
            'gemini-1.5-pro'        # Fallback
        ]
        
        last_error = None
        for model_name in model_names:
            try:
                model_gemini = genai.GenerativeModel(model_name)
                response = model_gemini.generate_content([prompt, img])
                clean_text = response.text.replace('**', '').replace('##', '')
                return clean_text
            except Exception as e:
                last_error = e
                # Continue to next model
                continue
        
        # If all models fail, provide helpful error message
        error_msg = str(last_error) if last_error else "No available Gemini models"
        raise Exception(f"Failed to connect to Gemini API. Tried models: {', '.join(model_names)}. Error: {error_msg}")
            
    except Exception as e:
        return f"Analysis could not be completed: {str(e)}. Please consult with a medical professional."

# ===============================
# PDF Generation
# ===============================
class NumberedCanvas:
    """Add page numbers to PDF"""
    def __init__(self, canvas, doc):
        self.canvas = canvas
        self.doc = doc
        
    def draw_page_number(self):
        self.canvas.saveState()
        self.canvas.setFont("Helvetica", 9)
        page_num = self.canvas.getPageNumber()
        text = f"Page {page_num}"
        self.canvas.drawCentredString(4.25*inch, 0.5*inch, text)
        self.canvas.restoreState()

def create_pdf_report(name, image_path, prediction_label, confidence, gemini_analysis, output_path):
    """Create PDF report with image and analysis"""
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    
    # Enhanced Styles
    styles = getSampleStyleSheet()
    
    # Title Style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=20,
        spaceBefore=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Subtitle Style
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#666666'),
        spaceAfter=25,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Section Heading Style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=5,
        backColor=colors.HexColor('#E3F2FD')
    )
    
    # Info Box Style
    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        leading=14,
        fontName='Helvetica'
    )
    
    # Normal Text Style
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#333333'),
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    # Bold Label Style
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1a237e'),
        fontName='Helvetica-Bold',
        spaceAfter=5
    )
    
    # Disclaimer Style
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#c62828'),
        leading=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica-Oblique',
        backColor=colors.HexColor('#FFEBEE'),
        borderWidth=1,
        borderColor=colors.HexColor('#c62828'),
        borderPadding=10
    )
    
    # Header Section
    header_table = Table([
        [Paragraph("🧠 Brain Tumor Detection Report", title_style)],
        [Paragraph("AI-Powered Medical Imaging Analysis", subtitle_style)]
    ], colWidths=[7*inch])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information Box
    patient_data = [
        ['Patient Name:', name],
        ['Report Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Report ID:', f"BT-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a237e')),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BBDEFB')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Detection Results Section
    story.append(Paragraph("📊 Detection Results", heading_style))
    
    # Results in a styled box
    condition_label = prediction_label.replace('_', ' ').title()
    confidence_color = colors.HexColor('#4CAF50') if confidence >= 70 else colors.HexColor('#FF9800') if confidence >= 50 else colors.HexColor('#F44336')
    
    results_data = [
        ['Predicted Condition:', condition_label],
        ['Confidence Level:', f"{confidence:.2f}%"],
        ['Status:', 'Preliminary Analysis']
    ]
    results_table = Table(results_data, colWidths=[2.5*inch, 4.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a237e')),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#283593')),
        ('TEXTCOLOR', (1, 1), (1, 1), confidence_color),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (1, 1), (1, 1), 12),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#FAFAFA'), colors.white]),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Analysis Visualization Section
    story.append(Paragraph("🔬 Analysis Visualization", heading_style))
    try:
        img = Image(image_path, width=5.5*inch, height=5.5*inch)
        img.hAlign = 'CENTER'
        story.append(img)
    except:
        story.append(Paragraph("<i>Image could not be loaded</i>", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # AI Medical Analysis Section
    story.append(Paragraph("🤖 AI Medical Analysis", heading_style))
    
    # Format analysis text with markdown parsing
    analysis_text = gemini_analysis.strip()
    
    def parse_markdown_to_html(text):
        """Convert markdown-style text to HTML for ReportLab"""
        import re
        
        # Replace markdown bold **text** with HTML <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Replace markdown italic *text* with HTML <i>text</i> (but not if it's a list)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        return text
    
    def format_analysis_text(text):
        """Format the analysis text into paragraphs with proper styling"""
        formatted_paragraphs = []
        
        # Split by double newlines to get main sections
        sections = text.split('\n\n')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            lines = section.split('\n')
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_paragraph:
                        formatted_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    continue
                
                # Check for main headings (lines with ** at start and end, or all caps short lines)
                if line.startswith('**') and line.endswith('**') and len(line) < 100:
                    if current_paragraph:
                        formatted_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    # Main heading
                    heading_text = parse_markdown_to_html(line)
                    formatted_paragraphs.append(('HEADING', heading_text))
                    continue
                
                # Check for numbered sections (1., 2., etc.)
                if re.match(r'^\d+[\.\)]\s+', line):
                    if current_paragraph:
                        formatted_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    # Numbered section heading
                    heading_text = parse_markdown_to_html(line)
                    formatted_paragraphs.append(('SUBHEADING', heading_text))
                    continue
                
                # Check for bullet points (* or -)
                if line.startswith('*') or line.startswith('-'):
                    if current_paragraph:
                        formatted_paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    # Bullet point
                    bullet_text = parse_markdown_to_html(line.lstrip('*- '))
                    formatted_paragraphs.append(('BULLET', bullet_text))
                    continue
                
                # Regular text line
                current_paragraph.append(line)
            
            if current_paragraph:
                formatted_paragraphs.append(' '.join(current_paragraph))
        
        return formatted_paragraphs
    
    # Format the analysis
    formatted_analysis = format_analysis_text(analysis_text)
    
    # Create styles for different elements
    subheading_style = ParagraphStyle(
        'SubheadingStyle',
        parent=normal_style,
        fontSize=12,
        textColor=colors.HexColor('#1976D2'),
        fontName='Helvetica-Bold',
        spaceAfter=8,
        spaceBefore=12,
        leftIndent=0.2*inch
    )
    
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=normal_style,
        fontSize=11,
        leftIndent=0.5*inch,
        bulletIndent=0.3*inch,
        spaceAfter=6,
        spaceBefore=4
    )
    
    # Render formatted paragraphs
    for para in formatted_analysis:
        if isinstance(para, tuple):
            para_type, para_text = para
            if para_type == 'HEADING':
                # Main heading
                story.append(Paragraph(para_text, subheading_style))
                story.append(Spacer(1, 0.1*inch))
            elif para_type == 'SUBHEADING':
                # Numbered section heading
                story.append(Paragraph(para_text, subheading_style))
                story.append(Spacer(1, 0.08*inch))
            elif para_type == 'BULLET':
                # Bullet point
                bullet_text = parse_markdown_to_html(para_text)
                story.append(Paragraph(f"• {bullet_text}", bullet_style))
        else:
            # Regular paragraph
            para_html = parse_markdown_to_html(para)
            story.append(Paragraph(para_html, normal_style))
            story.append(Spacer(1, 0.12*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Important Disclaimer Section
    story.append(Paragraph("⚠️ Important Disclaimer", heading_style))
    disclaimer_text = """
    <b>This report is generated by an AI system for preliminary analysis only.</b><br/><br/>
    It should not be used as a substitute for professional medical diagnosis, treatment, or advice. 
    The analysis provided is based on automated image processing and AI algorithms, which may have limitations 
    and should be interpreted with caution.<br/><br/>
    <b>Please consult with qualified healthcare professionals for accurate diagnosis and treatment recommendations.</b>
    """
    disclaimer_para = Paragraph(disclaimer_text, disclaimer_style)
    story.append(disclaimer_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Footer note
    footer_text = f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | AI-Powered Brain Tumor Detection System</i>"
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#999999'),
        alignment=TA_CENTER,
        spaceBefore=10
    )))
    
    # Build PDF with page numbers
    def on_first_page(canvas, doc):
        canvas.saveState()
        NumberedCanvas(canvas, doc).draw_page_number()
        canvas.restoreState()
    
    def on_later_pages(canvas, doc):
        canvas.saveState()
        NumberedCanvas(canvas, doc).draw_page_number()
        canvas.restoreState()
    
    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)

# ===============================
# Flask Routes
# ===============================
@app.route('/')
def index():
    """Home page with name input"""
    # Clear any old session data when starting fresh
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_name():
    """Store name in session and redirect to upload page"""
    name = request.form.get('name', '').strip()
    if not name:
        return render_template('index.html', error='Please enter your name')
    
    session['patient_name'] = name
    return render_template('upload.html', name=name)

@app.route('/process', methods=['POST'])
def process_image():
    """Process uploaded image"""
    if 'patient_name' not in session:
        return jsonify({'error': 'Session expired. Please start over.'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(upload_path)
        
        # Process brain scan
        output_image, label, confidence, predictions = process_brain_scan(upload_path)
        
        # Save output image
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{timestamp}_result.jpg")
        cv2.imwrite(output_image_path, output_image)
        
        # Get Gemini analysis
        gemini_analysis = analyze_tumor_stage(upload_path, label, confidence)
        
        # Create PDF report
        pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{timestamp}_report.pdf")
        create_pdf_report(
            session['patient_name'],
            output_image_path,
            label,
            confidence,
            gemini_analysis,
            pdf_path
        )
        
        # Store in session for download
        session['pdf_path'] = pdf_path
        session['output_image'] = output_image_path
        session['prediction'] = label
        session['confidence'] = confidence
        session['analysis'] = gemini_analysis
        
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': round(confidence, 2),
            'image_url': f'/output_image/{timestamp}_result.jpg',
            'pdf_url': f'/download_pdf/{timestamp}_report.pdf'
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/output_image/<filename>')
def output_image(filename):
    """Serve output images"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), mimetype='image/jpeg')

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    """Download PDF report"""
    pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=f"brain_tumor_report_{filename}")
    return "PDF not found", 404

@app.route('/result')
def result():
    """Display results page"""
    if 'patient_name' not in session:
        return render_template('index.html', error='Session expired. Please start over.')
    
    # Get image URL from session
    output_image = session.get('output_image', '')
    image_url = ''
    if output_image:
        filename = os.path.basename(output_image)
        image_url = f'/output_image/{filename}'
    
    # Get PDF URL from session
    pdf_path = session.get('pdf_path', '')
    pdf_url = ''
    if pdf_path:
        filename = os.path.basename(pdf_path)
        pdf_url = f'/download_pdf/{filename}'
    
    return render_template('result.html',
                         name=session.get('patient_name'),
                         prediction=session.get('prediction', 'N/A'),
                         confidence=session.get('confidence', 0),
                         analysis=session.get('analysis', ''),
                         image_url=image_url,
                         pdf_url=pdf_url)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)