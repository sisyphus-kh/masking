import os
import glob
import numpy as np
import pandas as pd
import cv2  # OpenCV for loading TIFF images
import torch  # PyTorch for loading the model and performing inference
import pickle  # To load the model from a .pkl file
import argparse  # For command-line argument parsing
from model import UNet
from osgeo import gdal, gdal_array

# Import RLE encoding and decoding functions from the separate file
from rle_encoder_decoder import rle_encode, rle_decode

def convert_to_nparr(image_path):
    dataset = gdal.Open(image_path)
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(dataset.GetRasterBand(1).DataType)
    arr = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount), dtype=dtype)
    bands = []
    for i in range(dataset.RasterCount):
        arr[:, :, i] = dataset.GetRasterBand(i + 1).ReadAsArray()
        bands.append(arr[:, :, i])
    bands = np.stack(bands)
    return bands

def load_image(image_path):
    img = convert_to_nparr(image_path)
    img = torch.from_numpy(img).float()  
    img = img.unsqueeze(0)  
    return img

def run_inference(model, image_paths, device):
    rle_masks = []
    total_dice = 0.0

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for image_path in image_paths:
            # Load and preprocess the image
            img_tensor = load_image(image_path).to(device)

            # Run inference
            pred_mask = model(img_tensor)

            # Post-process the predicted mask (thresholding)
            pred_mask = (pred_mask > 0.5).float()

            # RLE encode the mask
            rle_mask = rle_encode(pred_mask.squeeze().cpu().numpy())
            rle_masks.append(rle_mask)

            # Calculate Dice coefficient for evaluation
            dice = dice_coefficient(pred_mask, img_tensor)
            total_dice += dice

    test_dice = total_dice / len(image_paths)
    print(f"Test Dice Coefficient: {test_dice:.4f}")
    return rle_masks

def save_submission_file(rle_masks, image_names, output_file):
    df = pd.DataFrame({
        'id': image_names,
        'segmentation': rle_masks
    })
    df.to_csv(output_file, header=True, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test images and output RLE masks in a CSV file.")

    parser.add_argument(
        '--model', type=str, required=True, help="Path to the trained model file (e.g., 'best_model.pkl')."
    )
    
    parser.add_argument(
        '--test_dir', type=str, required=True, help="Path to the directory containing test TIFF images."
    )
    
    parser.add_argument(
        '--output', type=str, required=True, help="Path to the output CSV file (e.g., 'submission.csv')."
    )
    
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device to run inference on ('cpu' or 'cuda')."
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load the model from the .pkl file
    model = UNet(1)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    
    # Load the test image paths
    image_paths = glob.glob(os.path.join(args.test_dir, '*.tif'))
    image_names = [os.path.basename(f) for f in image_paths]
    
    # Run inference and get RLE-encoded masks
    rle_masks = run_inference(model, image_paths, args.device)
    
    # Save results to CSV
    save_submission_file(rle_masks, image_names, args.output)
    print(f"Submission file saved as {args.output}")

if __name__ == "__main__":
    main()

