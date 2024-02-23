"""
Download each DICOM, convert it into a 384x384 size PNG, 
and delete the original file. Requires 16T of bandwidth,
no need for 16T of storage space. 
Supports multiprocessing for downloading and processing. 
AWS configuration is required.
"""


import argparse
import boto3
import os
import pydicom
import numpy as np
from PIL import Image
from multiprocessing import Pool
import pandas as pd

# Argument parser setup
parser = argparse.ArgumentParser(description="Download and process DICOM files with options for image size and center cropping.")
parser.add_argument('--bucket_name', type=str, default="embed-dataset-open", help='The name of the S3 bucket.')
parser.add_argument('--local_directory', type=str, default="./EMBED/", help='Local directory path for downloading DICOM files.')
parser.add_argument('--size', type=int, default=384, help='Size of the output PNG image.')
parser.add_argument('--processes', type=int, default=100, help='Number of processes to use for file processing.')
parser.add_argument('--crop_center', action='store_true', help='Whether to perform center cropping on the image.')
args = parser.parse_args()

# Use parsed arguments
bucket_name = args.bucket_name
local_directory = args.local_directory
size = args.size
processes = args.processes
crop_center = args.crop_center

# Initialize the boto3 S3 client
s3 = boto3.client('s3')

class DCM_Tags():
    """Class to extract and store specific DICOM tags."""
    def __init__(self, img_dcm):
        try:
            self.laterality = img_dcm.ImageLaterality
        except AttributeError:
            self.laterality = np.nan
            
        try:
            self.view = img_dcm.ViewPosition
        except AttributeError:
            self.view = np.nan
            
        try:
            self.orientation = img_dcm.PatientOrientation
        except AttributeError:
            self.orientation = np.nan

# Check whether DICOM should be flipped
def check_dcm(imgdcm):
    # Get DICOM metadata
    tags = DCM_Tags(imgdcm)
    
    # If image orientation tag is defined
    if ~pd.isnull(tags.orientation):
        # CC view
        if tags.view == 'CC':
            if tags.orientation[0] == 'P':
                flipHorz = True
            else:
                flipHorz = False
            
            if (tags.laterality == 'L') & (tags.orientation[1] == 'L'):
                flipVert = True
            elif (tags.laterality == 'R') & (tags.orientation[1] == 'R'):
                flipVert = True
            else:
                flipVert = False
        
        # MLO or ML views
        elif (tags.view == 'MLO') | (tags.view == 'ML'):
            if tags.orientation[0] == 'P':
                flipHorz = True
            else:
                flipHorz = False
            
            if (tags.laterality == 'L') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HL')):
                flipVert = True
            elif (tags.laterality == 'R') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HR')):
                flipVert = True
            else:
                flipVert = False
        
        # Unrecognized view
        else:
            flipHorz = False
            flipVert = False
            
    # If image orientation tag is undefined
    else:
        # Flip RCC, RML, and RMLO images
        if (tags.laterality == 'R') & ((tags.view == 'CC') | (tags.view == 'ML') | (tags.view == 'MLO')):
            flipHorz = True
            flipVert = False
        else:
            flipHorz = False
            flipVert = False
            
    return flipHorz, flipVert

# Save DICOM pixel array as PNG
def save_dcm_image_as_png(image, png_filename, size=384):
    """Save DICOM pixel array as PNG with optional center cropping."""
    with open(png_filename, 'wb') as f:
        image = (image - image.min()) / (image.max() - image.min()) * 255
        img = Image.fromarray(image.astype('uint8'))
        scale = size / min(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        # 
        img_resized = img.resize(new_size)
        
        left = (img_resized.width - size) / 2
        top = (img_resized.height - size) / 2
        right = img_resized.width - left
        bottom = img_resized.height - top
        img_cropped = img_resized.crop((left, top, right, bottom))
        img_cropped.save(f)
        
# Convert list of DICOMs to PNGs
def process_dcm_list(dcm_list):
    for i, dcm_path in enumerate(dcm_list):
        print(f"Processing DICOM #{i}...")
        
        # Load DICOM
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array
        print("pixel got")
                
        # Check if a horizontal flip is necessary
        horz, _ = check_dcm(dcm)
        if horz:
            # Flip img horizontally
            img = np.fliplr(img)
        
        # Get new file name
        split_fn = dcm_path[:-4].split('/')
        new_fn = f"{split_fn[-1]}.png"
        
        # Save PNG
        png_path = os.path.join(os.path.join(*split_fn[:-1]),new_fn)
        save_dcm_image_as_png(img, png_path)


def download_and_process_dcm_file(key):
    """Download and process a single DICOM file."""
    print(f'Found DICOM file: {key}')
    download_path = os.path.join(local_directory, key.replace('/', os.sep))
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket_name, key, download_path)
    print(f'Downloaded: {key} to {download_path}')
    process_dcm_list([download_path])
    print("Processed")
    os.remove(download_path)

def download_dcm_files_multiprocess(bucket, processes=processes, prefix=''):
    """Use multiprocessing to download and process DICOM files."""
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        print("read page")
        dicom_keys = []
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(".dcm"):
                dicom_keys.append(key)
    

        with Pool(processes=processes) as pool:
            pool.map(download_and_process_dcm_file, dicom_keys)

if __name__ == '__main__':
    download_dcm_files_multiprocess(bucket_name, prefix='', processes=processes)
