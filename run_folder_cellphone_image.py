
import os
import datetime
import logging
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv


#Initialize configuration for Mask-RCNN
def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file("mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = "cpu" 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg.MODEL.WEIGHTS = 'model_final_01.pth' 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set the testing threshold for this model
    return cfg

def random_sat_color():
    """
    Generates random RGB triplet (0-255) for a saturated color
    Arg: None
    Output: trip - list of three numbers - one is 0, one is 255, and one is a
    random value (0-255)
    """
    trip = [0, 255, random.randint(0,255)]
    random.shuffle(trip)
    return trip

def draw_text_centered(text, img, loc, **kwargs):
    """
    Writes text to image at specific location
    Args:
        text - string to draw
        img - np.array image
        loc - center location to draw
        kwargs:
            fontFace, fontScale, thickness, bg_color, text_color
    """
    fontFace = kwargs.pop('fontFace', cv2.FONT_HERSHEY_SIMPLEX)
    fontScale = kwargs.pop('fontScale', 1.5)
    thickness = kwargs.pop('thickness', 3)
    bg_color = kwargs.pop('bg_color', (255, 255, 255))
    text_color = kwargs.pop('text_color', (0, 0, 0))

    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    textX_left = max(loc[0] - (textsize[0] // 2), 0)
    textY_bot = min(loc[1] + (textsize[1] // 2), np.shape(img)[0])

    textX_right = min(loc[0] + (textsize[0] // 2), np.shape(img)[1])
    textY_top = max(loc[1] - (textsize[1] // 2), 0)

    rectCorner1 = (textX_right, textY_bot)
    rectCorner2 = (textX_left, textY_top)

    img = cv2.rectangle(img, rectCorner1, rectCorner2, bg_color, -1)
    img = cv2.putText(img, text, (textX_left, textY_bot), fontFace, fontScale, text_color, thickness)

    return img

def get_sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield image[y:y + windowSize[1], x:x + windowSize[0]], y, x


def find_QR(img, retries=10, brute_force=False, debug=False):
    """
    Detect QR codes using OpenCV QRCodeDetector.
    Returns:
        data: Decoded string content of QR
        points: Corner points of QR
        y, x: shift values in case of sliding window (default 0, 0)
    """
    detector = cv2.QRCodeDetector()

    # Initial attempt to detect QR in the whole image
    data, points, _ = detector.detectAndDecode(img)

    if points is not None:
        points = points.reshape(-1, 2)

    if data and points is not None and len(points) >= 4:
        if debug:
            print(f"QR Code detected (full image): {data}")
            print(f"QR Code points: {points}")
        x, y = 0, 0
        return data, points, y, x

    if debug:
        print("QR Code not found in full image, attempting brute force..." if brute_force else "QR Code not found.")

    if brute_force:
        window_size = (1500, 1500)
        step_size = int(window_size[1] / 2)
    else:
        window_size = (img.shape[1], img.shape[0])
        step_size = img.shape[0]

    for (img_sub, y, x) in get_sliding_window(img, step_size, window_size):
        img_gray = cv2.cvtColor(img_sub, cv2.COLOR_BGR2GRAY)

        for i in range(retries):
            if debug:
                print(f"{i+1}th try on window at (x={x}, y={y})")

            blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            _, img_proc = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            data, points, _ = detector.detectAndDecode(img_proc)

            if points is not None:
                points = points.reshape(-1, 2)

            if data and points is not None and len(points) >= 4:
                if debug:
                    print(f"QR Code detected in window: {data}")
                    print(f"QR Code points: {points}")
                return data, points, y, x

    if debug:
        print("QR Code not found after brute force search.")
    return None


# --- Compute Calibrant ---
def compute_calibrant(data, points, y_shift, x_shift, qr_size=6, default_calibrant=90):
    """
    Compute calibrant value based on detected QR code points.
    """
    error_qr = 1
    pts_default = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    # Validate data and points
    if not data or points is None:
        print("QR Code not found!")
        return default_calibrant, pts_default, error_qr

    if len(points) < 2:
        print(f"Invalid QR points: expected at least 2 points, got {len(points)}")
        return default_calibrant, pts_default, error_qr

    # Calculate size in pixels from bounding box points
    qr_size_pixel = np.linalg.norm(points[0] - points[1])

    calibrant = qr_size_pixel / qr_size

    # Apply shifts to the points for global positioning
    pts = points + np.array([x_shift, y_shift])

    # Sanity check for calibrant value range
    if calibrant < 30 or calibrant > 160:
        print(f"Calibrant out of bounds: {calibrant}. Using default {default_calibrant}")
        return default_calibrant, pts_default, error_qr

    error_qr = 0
    return calibrant, pts, error_qr

def classify_sweetpot_class(weight_g, width_in, length_in):
    """
    classification function for specific categories: trash, canner, 90 count, 55 count, 40 count, 32 count, jumbo, oversize.

    Parameters:
    weight_g (float): Weight in grams.
    width_in (float): Width in inches.
    length_in (float): Length in inches.

    Returns:
    str: The classified category.
    """
    weight_oz = weight_g / 28.3495  # Convert grams to ounces

    if weight_oz <= 1 or width_in <= 1 or length_in <= 2:
        return "trash"
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
    elif 140 < weight_oz or width_in > 8.5 or length_in > 11:
        return "oversize"
    else:
        return "Unclassified"  # If no matching class is found

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}. Using blank image instead.")
        img = np.zeros((500, 500, 3), dtype=np.uint8)
    return img


def run_predictor(predictor, img, filename):
    try:
        outputs = predictor(img)
        predictions = outputs["instances"].to("cpu")
        masks = predictions.get('pred_masks').numpy().astype(np.uint8) * 255
        masks = np.moveaxis(masks, 0, -1)
        return masks
    except Exception as e:
        logging.error(f"Prediction error for {filename}: {e}")
        return None


def process_qr(img, filename, use_QR=True):
    if not use_QR:
        print("QR disabled. Using default calibrant.")
        logging.error(f"QR disabled. Using default calibrant for {filename}")
        return 90, None, 1
    try:
        QR_data = find_QR(img, retries=5, brute_force=True)
        if QR_data is None:
            print("No QR data found. Using default calibrant.")
            logging.error(f"No QR data found. Using default calibrant for {filename}")
            return 90, None, 1
        data, points, y_shift, x_shift = QR_data
        calibrant, pts, error_qr = compute_calibrant(data, points, y_shift, x_shift)
        if error_qr:
            print("QR calibrant out of range. Using default.")
            logging.error(f"QR calibrant out of range. Using default for {filename}")
            calibrant = 90
        return calibrant, pts, error_qr
    except Exception as e:
        logging.error(f"QR code processing error for {filename}: {e}")
        return 90, None, 1


def process_masks(masks, calibrant, filename, debug=False, use_QR=False, points=None, img=None):
    data_out = {
        'Filename': [],
        'Width (in)': [],
        'Length (in)': [],
        'Area (in^2)': [],
        'X Loc (px)': [],
        'Y Loc (px)': [],
        'Volume (in^3)': [],
        'Volume_alt (in^3)': [],
        'Solidity': [],
        'Strict_Solidity': [],
        'Weight (g)': [],
        'Weight_alt (g)': [],
        'Class': [],
    }

    for i in range(masks.shape[-1]):
        try:
            mask = masks[..., i]
            result = process_single_contour(mask, calibrant, i, debug, img)
            if result:
                for key in data_out.keys():
                    data_out[key].append(result[key])
        except Exception as e:
            logging.error(f"Error processing contour {i} in {filename}: {e}")
            print(f"Error processing contour {i}: {e}. Skipping contour...")

    return data_out


def process_single_contour(mask, calibrant, i, debug, img):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = contours[0]
    #measure demensions
    area = cv2.contourArea(cnt)
    if area == 0:
        return None

    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(cnt)
    center = [int(coord) for coord in rect[0]]
    width = min(rect[1])
    length = max(rect[1])
    #convert pixel to inches with calibration
    length_c = length / calibrant
    width_c = width / calibrant

    #check if QR code was detected as a potato
    if abs(length_c - width_c) < 0.1 and width_c > 6 and length_c > 6:
        print(f"Contour {i} skipped: potential QR code")
        return None
    
    perimeter = cv2.arcLength(cnt, True)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = area / hull_area if hull_area != 0 else 0
    strict_solidity = solidity * (hull_perimeter / perimeter) if perimeter != 0 else 0

    volume = (4/3) * (width/2) * area
    weight = 1.2 * 16.32 * (2/3) * area * (width/2) / calibrant ** 3

    sweetpot_class = classify_sweetpot_class(weight, width_c, length_c)

    #draw contour and text on image
    if debug and img is not None:
       draw_text_centered(str(i), img, center)
       cv2.drawContours(img, contours, -1, random_sat_color(), thickness=5)

    return {
        'Filename': "",
        'Width (in)': width_c,
        'Length (in)': length_c,
        'Area (in^2)': area / calibrant ** 2,
        'X Loc (px)': center[0],
        'Y Loc (px)': center[1],
        'Volume (in^3)': volume / calibrant ** 3,
        'Volume_alt (in^3)': (4/3) * 3.1415 * (width/2) * (width/2) * (length/2) / calibrant ** 3,
        'Solidity': solidity,
        'Strict_Solidity': strict_solidity,
        'Weight (g)': weight,
        'Weight_alt (g)': 10 ** (1.4444 * np.log10(area * (1/calibrant ** 2)) + 0.8142),
        'Class': sweetpot_class,
    }


def write_results(df, out_path, filename, img=None, write_out=True):
    output_csv_path = os.path.join(out_path, f'{filename}.csv')
    df.index.name = "Index"
    df.to_csv(output_csv_path, sep=',')

    if write_out and img is not None:
        output_img_path = os.path.join(out_path, f'{filename}.jpg')
        cv2.imwrite(output_img_path, cv2.resize(img, dsize=None, fx=1, fy=1))


def inference(predictor, img_path, out_path, debug=False, use_QR=True, calibrant=1, write_out=True):
    try:
        img = load_image(img_path)
        filename = os.path.basename(img_path).split('.')[0]

        masks = run_predictor(predictor, img, filename)
        if masks is None:
            return None

        calibrant, points, error_qr = process_qr(img, filename, use_QR)

        data_out = process_masks(masks, calibrant, filename, debug, use_QR, points, img)

        if not data_out['Filename']:
            logging.error(f"No valid detections for {img_path}")
            print(f"No valid detections for {img_path}.")
            return None

        df = pd.DataFrame(data_out)
        df['Calibrant (px/in)'] = calibrant
        df['Error QR (0 - no error, 1 - error)'] = error_qr

        write_results(df, out_path, filename, img, write_out)

        return data_out

    except Exception as e:
        logging.error(f"Unexpected error in inference for {img_path}: {e}")
        return None

def generate_aggregate_csv(directory):
    # Define the classes of objects to count
    classes = ["canner", "90 count", "55 count", "40 count", "32 count", "jumbo", "oversize", "trash", "Unclassified"]

    # Define the specific classes to be summed
    count_classes = ["90 count", "55 count", "40 count", "32 count"]

    # Initialize a dictionary to store aggregate data
    plot_data = {}
    # directory = os.path.join(os.getcwd(), "output") # Current directory/my_directory
    file_paths = [entry.path for entry in os.scandir(directory) if entry.is_file() and entry.name.endswith(".csv")]
    # Process each CSV file in the directory
    for file_path in file_paths:
        # Extract plot name from filename (everything before "_Image")
        filename = os.path.basename(file_path)
        plot_name = filename.split("_Image")[0]
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure necessary columns exist
        if "Weight (g)" not in df.columns or "Class" not in df.columns:
            continue  # Skip files without required columns
        
        # Initialize plot entry if not exists
        if plot_name not in plot_data:
            plot_data[plot_name] = {f"Weight of {cls}": 0 for cls in classes}  # Initialize weight sums
            plot_data[plot_name].update({f"Count of {cls}": 0 for cls in classes})  # Initialize counts
            plot_data[plot_name]["Count ones"] = 0  # Initialize combined count
            plot_data[plot_name]["Weight ones"] = 0  # Initialize combined weight
            plot_data[plot_name]["QR Flag"] = 0  # Initialize QR Flag
        # Aggregate sum of weights and counts for each class
        class_sums = df.groupby("Class")["Weight (g)"].sum()
        class_counts = df["Class"].value_counts()
        error = df["Error QR (0 - no error, 1 - error)"].sum()

        # Store the aggregated values in plot_data
        for cls in classes:
            if cls in class_sums:
                plot_data[plot_name][f"Weight of {cls}"] += class_sums[cls]
            if cls in class_counts:
                plot_data[plot_name][f"Count of {cls}"] += class_counts[cls]
        plot_data[plot_name]["QR Flag"] += error


        # Compute the combined values for "Count ones" and "Weight ones"
        plot_data[plot_name]["Count ones"] = sum(plot_data[plot_name][f"Count of {cls}"] for cls in count_classes)
        plot_data[plot_name]["Weight ones"] = sum(plot_data[plot_name][f"Weight of {cls}"] for cls in count_classes)

    # Convert the dictionary to a DataFrame
    output_df = pd.DataFrame.from_dict(plot_data, orient="index").reset_index()
    output_df.rename(columns={"index": "Plot name"}, inplace=True)
    for col in output_df.columns:
        if output_df[col].dtype == "float64":  # Check if column is a float
            output_df[col] = output_df[col].round(1)

    # Save to CSV
    output_csv_path = os.path.join(directory, "aggregated_results.csv")
    output_df.to_csv(output_csv_path, index=False)

def create_directory_if_not_exists(path):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(path):
        print(f"Directory does not exist. Creating: {path}")
        os.makedirs(path, exist_ok=True)


def get_image_files(input_path, img_suffixes=('png', 'jpg', 'jpeg')):
    """Get all image file paths in the directory matching the suffixes."""
    img_files = set()

    for file in os.listdir(input_path):
        fullpath = os.path.join(input_path, file)
        if os.path.isfile(fullpath) and fullpath.lower().endswith(img_suffixes):
            img_files.add(fullpath)
        # if len(img_files)>4:
        #     break

    return img_files


def process_new_images(predictor, new_imgs, output_path):
    """Run inference on new images."""
    for img_path in new_imgs:
        print(f"Processing image: {img_path}")
        _ = inference(predictor, img_path, output_path, debug=True, use_QR=True)


def main():
   
    # Load environment variables from the .env file
    load_dotenv(override=True)

    # Setup logging
    log_filename = f"errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = os.path.join(os.getenv('OUTPUT_DIR', '.'), log_filename)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    cfg = setup_model()
    # Initialize
    img_suffixes = ('png', 'jpg', 'jpeg')
    prev_img_names = set()

    # Paths from environment with fallback defaults
    input_path = os.getenv('INPUT_DIR')
    print(input_path)
    output_path = os.getenv('OUTPUT_DIR')
    print(output_path)
    # Ensure input/output directories exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    create_directory_if_not_exists(output_path)

    # Initialize predictor
    predictor = DefaultPredictor(cfg)

    # Collect image paths
    img_names = get_image_files(input_path, img_suffixes)

    # Detect new images (optional diff check)
    new_imgs = img_names - prev_img_names

    if new_imgs:
        process_new_images(predictor, new_imgs, output_path)

    # Aggregate CSV results
    generate_aggregate_csv(output_path)


if __name__ == "__main__":
    main()




