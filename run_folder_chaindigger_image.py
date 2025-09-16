import os
import datetime
import logging
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import random
from dotenv import load_dotenv

# --- Model Configuration ---
def setup_model(model):
    """Initializes and configures the Mask-RCNN model from Detectron2."""
    cfg = get_cfg()
    # Load model configuration from a YAML file
    cfg.merge_from_file("mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATALOADER.NUM_WORKERS = 2
    # Set the model to run on CPU
    cfg.MODEL.DEVICE = "cpu" 
    # There is only one class to detect (e.g., sweet potato)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    # Load the pre-trained model weights
    cfg.MODEL.WEIGHTS = model
    #cfg.MODEL.WEIGHTS = './model/chain_bst_20250820.pth' 
    # Set the confidence score threshold for detections
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    return cfg

# --- Image Annotation Helpers ---
def random_sat_color():
    """Generates a random saturated RGB color for drawing annotations."""
    trip = [0, 255, random.randint(0,255)]
    random.shuffle(trip)
    return tuple(trip)

def draw_text_centered(text, img, loc, **kwargs):
    """Writes text with a background rectangle onto an image at a centered location."""
    fontFace = kwargs.pop('fontFace', cv2.FONT_HERSHEY_SIMPLEX)
    fontScale = kwargs.pop('fontScale', 1.5)
    thickness = kwargs.pop('thickness', 3)
    bg_color = kwargs.pop('bg_color', (255, 255, 255))
    text_color = kwargs.pop('text_color', (0, 0, 0))

    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    
    # Calculate text box coordinates to be centered on 'loc'
    textX_left = max(loc[0] - (textsize[0] // 2), 0)
    textY_bot = min(loc[1] + (textsize[1] // 2), np.shape(img)[0])
    textX_right = min(loc[0] + (textsize[0] // 2), np.shape(img)[1])
    textY_top = max(loc[1] - (textsize[1] // 2), 0)

    # Draw background rectangle and then the text
    cv2.rectangle(img, (textX_left, textY_top), (textX_right, textY_bot), bg_color, -1)
    cv2.putText(img, text, (textX_left, textY_bot - 5), fontFace, fontScale, text_color, thickness)
    return img

# --- Timestamp and GPS Processing ---
def parse_image_filename_time(img_path_full: str) -> datetime | None:
    """
    Parses a datetime object from an image filename.
    Expected format: "...-YYYYMMDD_HHMMSS_mmm.jpg"
    """
    img_filename = os.path.basename(img_path_full)
    match = re.search(r"(\d{8})_(\d{6})_(\d{3})", img_filename)
    if match:
        date_str, time_seconds_str, ms_str = match.groups()
        try:
            timestamp_str = f"{date_str}{time_seconds_str}"
            base_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            img_dt = base_dt + timedelta(milliseconds=int(ms_str))
            return img_dt
        except ValueError as e:
            logging.error(f"Error parsing date/time from '{img_filename}': {e}")
            return None
    logging.warning(f"Timestamp not found in expected format in '{img_filename}'.")
    return None

def find_closest_gps_coords(pos_file_path: str, target_datetime_obj: datetime) -> tuple[float, float, float] | None:
    """Finds the GPS coordinates from a .pos file closest to a target datetime."""
    if target_datetime_obj is None:
        logging.error("Target datetime is None. Cannot find GPS coordinates.")
        return None
    try:
        # Read .pos file, skipping header lines. Using sep='\s+' instead of the deprecated delim_whitespace=True
        df = pd.read_csv(
            pos_file_path,
            comment='%',
            header=None,
            sep=r'\s+',
            usecols=[0, 1, 2, 3, 4],
            names=['date', 'time', 'latitude', 'longitude', 'height']
        )
        
        # Combine date and time columns and convert to datetime objects
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S.%f')
        
        # Calculate absolute time difference
        df['time_diff'] = (df['datetime'] - target_datetime_obj).abs()
        
        # Find the row with the minimum time difference
        closest_entry = df.loc[df['time_diff'].idxmin()]
        
        return closest_entry['longitude'], closest_entry['latitude'], closest_entry['height']

    except FileNotFoundError:
        logging.error(f"GPS file not found at '{pos_file_path}'")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing GPS file '{pos_file_path}': {e}")
        return None
    
def find_closest_gps_coords_interp(pos_file_path: str, target_datetime_obj: datetime) -> tuple[float, float, float] | None:
    """
    Finds the GPS coordinates from a .pos file closest to a target datetime,
    with linear interpolation.
    """
    if target_datetime_obj is None:
        logging.error("Target datetime is None. Cannot find GPS coordinates.")
        return None
    try:
        # Read .pos file, skipping header lines. Using sep='\s+' instead of the deprecated delim_whitespace=True
        df = pd.read_csv(
            pos_file_path,
            comment='%',
            header=None,
            sep=r'\s+',
            usecols=[0, 1, 2, 3, 4],
            names=['date', 'time', 'latitude', 'longitude', 'height']
        )
        
        # Combine date and time columns and convert to datetime objects
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M:%S.%f')
        df = df.sort_values(by='datetime').reset_index(drop=True)

        if df.empty:
            logging.warning("GPS DataFrame is empty. Cannot find GPS coordinates.")
            return None

        # Log GPS data range and target image timestamp for debugging
        logging.info(f"GPS data range: From {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        logging.info(f"Image target timestamp: {target_datetime_obj}")

        # Use searchsorted to find the insertion point
        # This will return the index where target_datetime_obj would be inserted to maintain order.
        # If target_datetime_obj is before the first element, idx_after will be 0.
        # If target_datetime_obj is after the last element, idx_after will be len(df).
        idx_after = df['datetime'].searchsorted(target_datetime_obj)
        
        # Case 1: Target is before the first GPS point
        if idx_after == 0:
            closest_entry = df.iloc[0]
            logging.warning(f"Image timestamp {target_datetime_obj} is before the first GPS point ({df['datetime'].iloc[0]}). Using first GPS point without interpolation.")
            return closest_entry['longitude'], closest_entry['latitude'], closest_entry['height']
        
        # Case 2: Target is after the last GPS point
        elif idx_after == len(df):
            closest_entry = df.iloc[-1]
            logging.warning(f"Image timestamp {target_datetime_obj} is after the last GPS point ({df['datetime'].iloc[-1]}). Using last GPS point without interpolation.")
            return closest_entry['longitude'], closest_entry['latitude'], closest_entry['height']
        
        # Case 3: Target is within the range, interpolate
        else:
            idx_before = idx_after - 1
            
            p1 = df.iloc[idx_before]
            p2 = df.iloc[idx_after]

            t1 = p1['datetime']
            t2 = p2['datetime']

            # Calculate interpolation factor
            # Use timedelta objects for more robust time difference calculation
            time_diff_total = (t2 - t1).total_seconds()
            if time_diff_total == 0:
                logging.warning(f"Duplicate GPS timestamps found at {p1['datetime']}. Cannot interpolate, using point before.")
                return p1['longitude'], p1['latitude'], p1['height']

            alpha = (target_datetime_obj - t1).total_seconds() / time_diff_total

            interp_longitude = p1['longitude'] + alpha * (p2['longitude'] - p1['longitude'])
            interp_latitude = p1['latitude'] + alpha * (p2['latitude'] - p1['latitude'])
            interp_height = p1['height'] + alpha * (p2['height'] - p1['height'])
            
            logging.info(f"Interpolated GPS for image timestamp {target_datetime_obj}: Long={interp_longitude:.6f}, Lat={interp_latitude:.6f}, H={interp_height:.3f}")
            return interp_longitude, interp_latitude, interp_height

    except FileNotFoundError:
        logging.error(f"GPS file not found at '{pos_file_path}'")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing GPS file '{pos_file_path}': {e}")
        return None

# --- Object Classification and Measurement ---
def classify_sweetpot_class(weight_g, width_in, length_in):
    """Classifies a sweet potato based on its weight, width, and length."""
    weight_oz = weight_g / 28.3495  # Convert grams to ounces

    if weight_oz <= 1 or width_in <= 1 or length_in <= 2:
        return "trash"
    elif length_in/width_in > 4.5:
        return "lwg4p5"
    elif 1 < weight_oz < 5 or width_in < 2 or length_in < 3:
        return "canner"
    elif 5 <= weight_oz <= 9.4 and 2 <= width_in <= 3.5 and 3 <= length_in <= 9:
        return "90 count"
    elif 9.4 < weight_oz <= 14 and 2 <= width_in <= 3.5 and 3 <= length_in <= 9:
        return "55 count"
    elif 14 < weight_oz <= 18 and 2 <= width_in <= 3.5 and 3 <= length_in <= 9:
        return "40 count"
    elif 18 < weight_oz <= 22 and 2 <= width_in <= 3.5 and 3 <= length_in <= 9:
        return "32 count"
    elif 22 < weight_oz <= 140 or 3.5 < width_in <= 8.5 or 9 < length_in <= 11:
        return "jumbo"
    elif weight_oz > 140 or width_in > 8.5 or length_in > 11:
        return "oversize"
    else:
        return "Unclassified"

def process_single_contour(mask, calibrant, index, img_to_draw_on=None):
    """Processes a single mask to extract measurements and classify the object."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea) # Use the largest contour
    area_px = cv2.contourArea(cnt)
    if area_px < 100:  # Ignore very small detections
        return None

    # --- Measurements ---
    rect = cv2.minAreaRect(cnt)
    center_px = tuple(map(int, rect[0]))
    width_px = min(rect[1])
    length_px = max(rect[1])
    
    # Convert from pixels to inches using the calibrant
    length_in = length_px / calibrant
    width_in = width_px / calibrant
    area_in2 = area_px / (calibrant ** 2)

    # --- Shape Properties ---
    hull = cv2.convexHull(cnt)
    hull_area_px = cv2.contourArea(hull)
    solidity = area_px / hull_area_px if hull_area_px > 0 else 0

    # --- Volume & Weight Estimation ---
    # Simplified volume/weight estimations
    #volume_in3 = (4/3) * np.pi * (width_in/2)**2 * (length_in/2) # Ellipsoid volume
    #ELlipsoid volume from known area
    volume_in3 = (4/3) * area_in2 * (width_in / 2)
    # Density assumption would be needed for accurate weight. This is a placeholder.
    # Specific gravity of sweetpotato is ~1.05 g/cm^3. 1 in^3 = 16.3871 cm^3
    weight_g = volume_in3 * 16.3871 * 1.05
    weigth_g_alt =(10 ** (1.4444 * np.log10(area_px * (1/calibrant ** 2)) + 0.8142))

    sweetpot_class = classify_sweetpot_class(weight_g, width_in, length_in)

    # --- Annotation ---
    if img_to_draw_on is not None:
       draw_text_centered(str(index), img_to_draw_on, center_px)
       cv2.drawContours(img_to_draw_on, [cnt], -1, random_sat_color(), thickness=3)

    return {
        'Width (in)': width_in,
        'Length (in)': length_in,
        'Area (in^2)': area_in2,
        'X Loc (px)': center_px[0],
        'Y Loc (px)': center_px[1],
        'Volume (in^3)': volume_in3,
        'Solidity': solidity,
        'Weight (g)': weight_g,
        'Weight Alt (g)': weigth_g_alt,
        'Class': sweetpot_class,
    }

# --- Main Processing Functions ---
def process_image(predictor, img_path, out_path, calibrant, gps_file, write_out=True):
    """
    Main function to process a single image: load, predict, measure, and save results.
    """
    logging.info(f"Processing image: {os.path.basename(img_path)}")
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"Failed to load image: {img_path}")
        return

    img_for_annotation = img.copy() if write_out else None
    filename_base = os.path.splitext(os.path.basename(img_path))[0]
    
    # --- Run Inference ---
    try:
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")
        masks = predictions.get('pred_masks').numpy()
    except Exception as e:
        logging.error(f"Prediction failed for {img_path}: {e}")
        return

    # --- Process each detected instance ---
    results_list = []
    for i in range(len(masks)):
        mask = (masks[i] * 255).astype(np.uint8)
        result = process_single_contour(mask, calibrant, i, img_for_annotation)
        if result:
            results_list.append(result)

    if not results_list:
        logging.warning(f"No valid detections in {img_path}.")
        return

    # --- Create DataFrame and add metadata ---
    df = pd.DataFrame(results_list)
    df['Filename'] = os.path.basename(img_path)
    df['Calibrant (px/in)'] = calibrant
    
    # --- Add GPS Data ---
    img_time = parse_image_filename_time(img_path)
    if gps_file and img_time:
        coords = find_closest_gps_coords_interp(gps_file, img_time)
        if coords:
            df['Longitude'], df['Latitude'], df['Height'] = coords
    
    # --- Write Results ---
    if write_out:
        output_csv_path = os.path.join(out_path, f'{filename_base}.csv')
        df.index.name = "Index"
        df.to_csv(output_csv_path)

        output_img_path = os.path.join(out_path, f'{filename_base}_annotated.jpg')
        cv2.imwrite(output_img_path, img_for_annotation)

def concatenate_all_results(directory):
    """Concatenates all individual result CSVs into a single master CSV file."""
    all_csvs = [f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('all_') and not f.startswith('aggregated_')]
    if not all_csvs:
        logging.info("No individual CSV files found to concatenate.")
        return

    df_list = []
    for f in all_csvs:
        try:
            df = pd.read_csv(os.path.join(directory, f))
            df_list.append(df)
        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping empty CSV file: {f}")
            continue
    
    if not df_list:
        logging.warning("All individual CSVs were empty. No concatenated file will be generated.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    
    output_csv_path = os.path.join(directory, "all_results.csv")
    combined_df.to_csv(output_csv_path, index=False)
    logging.info(f"All results concatenated and saved to {output_csv_path}")

# --- Utility Functions ---
def get_files(directory, extensions):
    """Gets all files in a directory with given extensions."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]

def main():
    """Main execution function."""
    # Load environment variables (e.g., for INPUT_DIR, OUTPUT_DIR, CALIBRANT)
    load_dotenv(override=True)

    input_path = os.getenv('INPUT_DIR', './input')
    output_path = os.getenv('OUTPUT_DIR', './output')
    model = os.getenv('MODEL_PATH', './model/20250908_model.pth')
    # Calibrant: pixels per inch. Must be measured from a reference object.
    calibrant = float(os.getenv('CALIBRANT', '100.0')) 

    # --- Setup Output Directory and Logging ---
    os.makedirs(output_path, exist_ok=True)
    
    log_file_path = os.path.join(output_path, 'processing_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
            logging.StreamHandler() # Also print logs to console
        ]
    )
    
    if not os.path.exists(input_path):
        logging.critical(f"Input directory does not exist: {input_path}")
        return

    # --- Find GPS file ---
    pos_files = get_files(input_path, ('.pos'))
    gps_file = pos_files[0] if pos_files else None
    if gps_file:
        logging.info(f"Using GPS position file: {os.path.basename(gps_file)}")
    else:
        logging.warning("No .pos GPS file found in the input directory.")

    # --- Initialize Model and Find Images ---
    cfg = setup_model(model)
    predictor = DefaultPredictor(cfg)
    img_paths = get_files(input_path, ('.png', '.jpg', '.jpeg'))
    
    if not img_paths:
        logging.warning(f"No images found in {input_path}")
        return
        
    # --- Process each image ---
    for img_path in img_paths:
        process_image(predictor, img_path, output_path, calibrant, gps_file, write_out=True)

    # --- Final Reporting ---
    logging.info("Concatenating all result files...")
    concatenate_all_results(output_path)
    
    logging.info("Processing complete. Check the output folder for results and logs.")


if __name__ == "__main__":
    main()
