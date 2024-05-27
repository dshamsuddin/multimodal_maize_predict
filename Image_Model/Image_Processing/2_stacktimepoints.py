import os
import numpy as np
from PIL import Image

def is_image(filename):
    img_extensions = ['.jpg']
    return any(filename.lower().endswith(ext) for ext in img_extensions)

def stack_similar_images(input_dir, output_dir):
    # Create a list to store pairs of stacked images and their contributing images
    stacked_images_list = []
    
    # Create a list to store the names of images that couldn't be stacked
    failed_to_stack = []
    
    # Recursively get all subdirectories and files within the input directory
    for root, _, filenames in os.walk(input_dir):
        # Filter out non-image files
        image_filenames = [filename for filename in filenames if is_image(filename)]
        
        # Create a dictionary to store the image paths based on the common part of their filenames
        image_dict = {}
        
        # Loop through the filenames and group the image paths based on the common part of their filenames
        for filename in image_filenames:
            common_name = os.path.splitext(filename[6:])[0]  # Remove the first six characters
            if common_name not in image_dict:
                image_dict[common_name] = []
            image_dict[common_name].append(os.path.join(root, filename))
        
        # Loop through the grouped image paths and stack the images with similar filenames
        for common_name, image_paths in image_dict.items():
            # Ensure that only 2 images are stacked
            if len(image_paths) == 2:
                # Extract the dates from the image filenames
                dates = [os.path.splitext(os.path.basename(img_path))[0][:6] for img_path in image_paths]
                
                # Extract the MM (month) part of the filenames and convert them to integers
                months = [int(date[2:4]) for date in dates]
                
                # Sort the image paths by MM in ascending order
                sorted_image_paths = [x for _, x in sorted(zip(months, image_paths))]
                
                stacked_images = [np.array(Image.open(img_path)) for img_path in sorted_image_paths]

                # Find the dimensions of the largest image
                max_width = max(img.shape[1] for img in stacked_images)
                max_height = max(img.shape[0] for img in stacked_images)

                # Create new images with padding to match the largest dimensions
                padded_images = []
                for img in stacked_images:
                    pad_width = max_width - img.shape[1]
                    pad_height = max_height - img.shape[0]
                    padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                    padded_images.append(padded_img)

                stacked_image = np.vstack(padded_images)
                stacked_image = Image.fromarray(stacked_image)
                
                # Save the stacked image in the new subdirectory within the output directory
                relative_path = os.path.relpath(root, input_dir)
                output_subdirectory = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdirectory, exist_ok=True)
                
                # Create a new filename for the stacked image by using the common name and sorted dates
                new_filename = f"{common_name}_stacked_{dates[0]}_{dates[1]}.jpg"
                
                # Save the stacked image
                stacked_image_path = os.path.join(output_subdirectory, new_filename)
                stacked_image.save(stacked_image_path)
                
                # Store the pair of stacked image path and its contributing image paths in the list
                stacked_images_list.append((stacked_image_path, image_paths))
            else:
                # If there are more or fewer than 2 images, add them to the failed to stack list
                failed_to_stack.extend(image_paths)

    # Save the list of pairs of stacked images and their contributing images in the text file
    with open(os.path.join(output_dir, "stacked_images_list.txt"), "w") as f:
        for stacked_image_path, contributing_images in stacked_images_list:
            f.write(f"Stacked Image: {stacked_image_path}\n")
            f.write("Contributing Images:\n")
            for img_path in contributing_images:
                f.write(f"{img_path}\n")
            f.write("\n")  # Add an empty line as a separator between stacked images
        
        # Check if any images failed to stack and write a message accordingly
        if failed_to_stack:
            f.write("Errors were encountered during stacking for the following images:\n")
            for img_path in failed_to_stack:
                f.write(f"{img_path}\n")
        else:
            f.write("No errors were encountered during stacking.\n")


# Get the current directory
current_directory = os.getcwd()

# Define the output directories
stacked_images_directory = os.path.join(current_directory, "stacked_timepoints")

# Create output directories if they don't exist
os.makedirs(stacked_images_directory, exist_ok=True)

stack_similar_images(current_directory, stacked_images_directory)
