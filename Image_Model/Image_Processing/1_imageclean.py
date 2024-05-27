import os
from PIL import Image

# Function to rotate images
def rotate_images(input_directory, output_directory):
    # Get a list of all .tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith(".tif")]

    # Process each .tif file
    for tif_file in tif_files:
        image_path = os.path.join(input_directory, tif_file)
        image = Image.open(image_path)
        rotated_image = image.rotate(131, expand=True)

        base_name = os.path.basename(image_path)
        output_name = os.path.splitext(base_name)[0] + "_rotated.tif"
        output_path = os.path.join(output_directory, output_name)

        rotated_image.save(output_path)

# Function to crop invisible data pixels
def crop_invisible_data_pixels(input_directory, output_directory):
    # Get a list of all .tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith("_rotated.tif")]

    # Process each rotated .tif file
    for tif_file in tif_files:
        image_path = os.path.join(input_directory, tif_file)
        image = Image.open(image_path)

        image_rgba = image.convert("RGBA")
        red, green, blue, alpha = image_rgba.split()
        bbox = alpha.getbbox()
        cropped_image = image.crop(bbox)

        base_name = os.path.basename(image_path)
        output_name = os.path.splitext(base_name)[0] + "_clean.tif"
        output_path = os.path.join(output_directory, output_name)

        cropped_image.save(output_path)

# Function to split images into quadrants
def split_images_into_quadrants(input_directory, output_directory):
    # Get a list of all .tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith("_clean.tif")]

    # Process each cleaned .tif file
    for tif_file in tif_files:
        image_path = os.path.join(input_directory, tif_file)
        image = Image.open(image_path)

        width, height = image.size
        quadrant_width = width // 2
        quadrant_height = height // 2

        top_left = image.crop((0, 0, quadrant_width, quadrant_height))
        top_right = image.crop((quadrant_width, 0, width, quadrant_height))
        bottom_left = image.crop((0, quadrant_height, quadrant_width, height))
        bottom_right = image.crop((quadrant_width, quadrant_height, width, height))

        base_name = os.path.basename(image_path)
        output_name = os.path.splitext(base_name)[0]

        top_left.save(os.path.join(output_directory, output_name + "_quadrant_1.tif"))
        top_right.save(os.path.join(output_directory, output_name + "_quadrant_2.tif"))
        bottom_left.save(os.path.join(output_directory, output_name + "_quadrant_3.tif"))
        bottom_right.save(os.path.join(output_directory, output_name + "_quadrant_4.tif"))

# Function to stack quadrant files
def stack_quadrant_files(input_directory, output_directory):
    # Get a list of all _quadrant_1.tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith("quadrant_1.tif")]

    # Process each _quadrant_1.tif file
    for tif_file in tif_files:
        image_name = tif_file[:-14]
        quadrant_1_path = os.path.join(input_directory, tif_file)
        quadrant_2_path = os.path.join(input_directory, image_name + "quadrant_2.tif")
        quadrant_3_path = os.path.join(input_directory, image_name + "quadrant_3.tif")
        quadrant_4_path = os.path.join(input_directory, image_name + "quadrant_4.tif")

        quadrant_1 = Image.open(quadrant_1_path)
        quadrant_2 = Image.open(quadrant_2_path)
        quadrant_3 = Image.open(quadrant_3_path)
        quadrant_4 = Image.open(quadrant_4_path)

        # Stack quadrant files 1 and 2
        stacked_1_2 = Image.new("RGB", (quadrant_1.width, quadrant_1.height * 2))
        stacked_1_2.paste(quadrant_1, (0, 0))
        stacked_1_2.paste(quadrant_2, (0, quadrant_1.height))

        # Stack quadrant files 3 and 4
        stacked_3_4 = Image.new("RGB", (quadrant_3.width, quadrant_3.height * 2))
        stacked_3_4.paste(quadrant_3, (0, 0))
        stacked_3_4.paste(quadrant_4, (0, quadrant_3.height))

        # Save stacked images
        stacked_1_2.save(os.path.join(output_directory, image_name + "_stacked_1_2.tif"))
        stacked_3_4.save(os.path.join(output_directory, image_name + "_stacked_3_4.tif"))

# Get the current directory
current_directory = os.getcwd()

# Define the output directories
rotated_images_directory = os.path.join(current_directory, "rotated_images")
cleaned_images_directory = os.path.join(current_directory, "cleaned_images")
quadrants_directory = os.path.join(current_directory, "chunks")
output_directory = os.path.join(current_directory, "stacked_images")

# Create output directories if they don't exist
os.makedirs(rotated_images_directory, exist_ok=True)
os.makedirs(cleaned_images_directory, exist_ok=True)
os.makedirs(quadrants_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Step 1: Rotate images
rotate_images(current_directory, rotated_images_directory)

# Step 2: Crop invisible data pixels
crop_invisible_data_pixels(rotated_images_directory, cleaned_images_directory)

# Step 3: Split images into quadrants
split_images_into_quadrants(cleaned_images_directory, quadrants_directory)

# Step 4: Stack quadrant files 1 and 2, and quadrant files 3 and 4
stack_quadrant_files(quadrants_directory, output_directory)
