import os
from PIL import Image

def convert_tif_to_jpg():
    current_directory = os.getcwd()
    tif_folder_name = "all_images"
    jpg_folder_name = "all_images_jpg"
    tif_folder_path = os.path.join(current_directory, tif_folder_name)
    jpg_folder_path = os.path.join(current_directory, jpg_folder_name)

    # Create a new folder for JPG images if it doesn't exist
    if not os.path.exists(jpg_folder_path):
        os.makedirs(jpg_folder_path)

    # Get a list of all .tif files in the tif_folder
    tif_files = [file for file in os.listdir(tif_folder_path) if file.endswith('.tif')]

    # Convert .tif files to .jpg and save them in the jpg_folder
    for file in tif_files:
        tif_path = os.path.join(tif_folder_path, file)
        img = Image.open(tif_path)
        jpg_path = os.path.join(jpg_folder_path, os.path.splitext(file)[0] + ".jpg")
        img.save(jpg_path, "JPEG")

if __name__ == "__main__":
    convert_tif_to_jpg()
