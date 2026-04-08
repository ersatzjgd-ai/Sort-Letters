import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from rapidfuzz import process, fuzz
import re
import cv2  # Added for blur/clarity detection

# --- UI Styling ---
st.set_page_config(page_title="Letter Sorter", layout="centered")

st.markdown("""
    <style>
    .big-poc-card {
        background-color: #198754; color: white; padding: 40px 20px;
        border-radius: 12px; text-align: center; font-size: 50px;
        font-weight: 900; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 10px;
    }
    .match-text { color: #495057; font-size: 20px; text-align: center; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

st.title("Sort letters for POCs")
st.write("Scan the letterhead to match it with its POC. The app will automatically adjust to standard column naming conventions.")

# --- 1. Data Loading ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload POC List (CSV/Excel)", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        
        if 'Name' not in df.columns:
            st.error("Error: The file must contain a 'Name' column for the invitees.")
            return None
            
        df = df.dropna(subset=['Name'])
        
        # --- Handle POC Column Variations ---
        poc_variations = ['Main POC Name', 'Main POC Names', 'POC Name', 'POC Names', 'POC']
        existing_poc_col = next((col for col in poc_variations if col in df.columns), None)
        
        if existing_poc_col:
            # Standardize the column name to 'Main POC Name' internally
            df.rename(columns={existing_poc_col: 'Main POC Name'}, inplace=True)
        else:
            st.error(f"Error: Could not find a POC column. Please use one of: {', '.join(poc_variations)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if not uploaded_file:
    st.info("Upload your contact list to start.")
    st.stop()

df = load_data(uploaded_file)
if df is None: st.stop()
invitee_list = df['Name'].astype(str).tolist()

# --- 2. Smart Name Scrubber & Image Checking Functions ---
def clean_scanned_text(text):
    """Removes common titles and suffixes to isolate the actual name."""
    noise_words = [
        r'\bMr\b', r'\bMs\b', r'\bMrs\b', r'\bShri\b', r'\bSmt\b', 
        r'\bDr\b', r'\bProf\b', r'\bJi\b', r'\bHon\b', r'\bCapt\b', r'\bLate\b'
    ]
    cleaned = text
    for word in noise_words:
        cleaned = re.sub(word, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    return cleaned.strip()

def is_image_clear(image_array, threshold=100.0):
    """Calculates the Laplacian variance to determine if the image is blurry."""
    # Convert RGB (from PIL) to Grayscale for OpenCV
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Calculate the variance of the Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # If variance is below the threshold, it's likely blurry
    return variance >= threshold, variance

# --- 3. AI Model Initialization ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# --- 4. Capture & Matching Logic ---
captured_image = st.camera_input("Snap a photo of the label", label_visibility="collapsed")

if captured_image:
    img = Image.open(captured_image)
    img_array = np.array(img)
    
    # Check image clarity before running heavy OCR
    is_clear, focus_score = is_image_clear(img_array)
    
    if not is_clear:
        st.warning(f"⚠️ **Image is too blurry or unstable!** (Clarity score: {focus_score:.1f}). Please hold the camera still and snap a clearer photo.")
    else:
        with st.spinner("Image is clear! AI is scrubbing and matching names..."):
            results = reader.readtext(img_array)
            
            # Filter for "real" text
            valid_lines = [res[1].strip() for res in results if len(res[1].strip()) > 2]
            
            # Check first two lines
            lines_to_check = valid_lines[:2] 
            
            match_found = False
            best_overall_match = None
            highest_score = 0

            for original_line in lines_to_check:
                cleaned_line = clean_scanned_text(original_line)
                match = process.extractOne(cleaned_line, invitee_list, scorer=fuzz.token_set_ratio)
                
                if match and match[1] > 70: # Confidence threshold
                    if match[1] > highest_score:
                        highest_score = match[1]
                        best_overall_match = (match[0], original_line)

            # --- 5. Results ---
            st.markdown("---")
            if best_overall_match:
                final_name, raw_scanned = best_overall_match
                poc = df.loc[df['Name'] == final_name, 'Main POC Name'].values[0]
                
                st.markdown(f'<div class="big-poc-card">{poc}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="match-text">Matched: <b>{final_name}</b></div>', unsafe_allow_html=True)
                st.caption(f"Scanner read: '{raw_scanned}' (Titles ignored)")
            else:
                st.error("⚠️ No match found.")
                with st.expander("Diagnostic: What the AI detected"):
                    st.write("Lines detected:", lines_to_check)
                    st.write("Cleaned lines (after removing titles):", [clean_scanned_text(l) for l in lines_to_check])

st.sidebar.write(f"Database: {len(df)} names loaded.")
