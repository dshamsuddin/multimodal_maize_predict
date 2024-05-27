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


def crop_images_in_directory(input_dir, output_dir, pixels_to_remove):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    image_files = [file for file in os.listdir(input_dir) if file.lower().endswith(('.tif'))]

    for filename in image_files:
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        # Open the image
        img = Image.open(input_image_path)

        # Get the current dimensions
        width, height = img.size

        # Calculate new dimensions after removing pixels from the right side
        new_width = width - pixels_to_remove
        new_height = height

        # Crop the image
        cropped_img = img.crop((0, 0, new_width, new_height))

        # Save the cropped image to the output directory
        cropped_img.save(output_image_path)

# Define the number of pixels to remove
pixels_to_remove = 22


# Function to split images into quadrants
def split_images_into_quadrants(input_directory, output_directory):
    # Get a list of all .tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith("_clean.tif")]

    # Process each cleaned .tif file
    for tif_file in tif_files:
        image_path = os.path.join(input_directory, tif_file)
        image = Image.open(image_path)

        width, height = image.size

        # Split the image into left and right halves
        half_width = width // 2
        left_half = image.crop((0, 0, half_width, height))
        right_half = image.crop((half_width, 0, width, height))

        base_name = os.path.basename(image_path)
        output_name = os.path.splitext(base_name)[0]

        # Save the halves as separate images
        left_half.save(os.path.join(output_directory, output_name + "_quadrant_1.tif"))
        right_half.save(os.path.join(output_directory, output_name + "_quadrant_2.tif"))


# Function to stack quadrant files
def stack_quadrant_files(input_directory, output_directory):
    # Get a list of all _quadrant_1.tif files in the input directory
    tif_files = [file for file in os.listdir(input_directory) if file.endswith("quadrant_1.tif")]

    # Process each _quadrant_1.tif file
    for tif_file in tif_files:
        image_name = tif_file[:-14]
        quadrant_1_path = os.path.join(input_directory, tif_file)
        quadrant_2_path = os.path.join(input_directory, image_name + "quadrant_2.tif")

        quadrant_1 = Image.open(quadrant_1_path)
        quadrant_2 = Image.open(quadrant_2_path)

        # Stack quadrant files 1 and 2
        stacked_1_2 = Image.new("RGB", (quadrant_1.width, quadrant_1.height * 2))
        stacked_1_2.paste(quadrant_1, (0, 0))
        stacked_1_2.paste(quadrant_2, (0, quadrant_1.height))

        # Save stacked images
        stacked_1_2.save(os.path.join(output_directory, image_name + "_stacked_1_2.tif"))


# Get the current directory
current_directory = os.getcwd()


# Define the output directories
rotated_images_directory = os.path.join(current_directory, "rotated_images")
cleaned_images_directory = os.path.join(current_directory, "cleaned_images")
cropped_directory = os.path.join(current_directory, "cropped_images")
quadrants_directory = os.path.join(current_directory, "chunks")
output_directory = os.path.join(current_directory, "stacked_images")

# Create output directories if they don't exist
os.makedirs(rotated_images_directory, exist_ok=True)
os.makedirs(cleaned_images_directory, exist_ok=True)
os.makedirs(cropped_directory, exist_ok=True)
os.makedirs(quadrants_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Step 1: Rotate images
rotate_images(current_directory, rotated_images_directory)

# Step 2: Crop invisible data pixels
crop_invisible_data_pixels(rotated_images_directory, cleaned_images_directory)

# Step 3: Crop images in the current directory
crop_images_in_directory(cleaned_images_directory, cropped_directory, pixels_to_remove)

# Step 4: Split images in half
split_images_into_quadrants(cropped_directory, quadrants_directory)

# Step 5: Stack quadrant files 1 and 2
stack_quadrant_files(quadrants_directory, output_directory)








