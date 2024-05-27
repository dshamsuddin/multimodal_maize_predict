from fastai.vision.all import *
import fastai
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
from pathlib import Path

# CUSTOM VIS DATABLOCK FUNCTIONS

# +
import os
import numpy as np
from PIL import Image

def is_image(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(filename.lower().endswith(ext) for ext in img_extensions)

def find_max_dimensions(input_dir):
    max_width = 0
    max_height = 0
    max_image = None
    
    # Get a list of all .jpg image filenames in the input directory
    filenames = [filename for filename in os.listdir(input_dir) if is_image(filename)]

    for filename in filenames:
        img = Image.open(os.path.join(input_dir, filename))
        img_width, img_height = img.size

        # Update max_width and max_height if needed
        if img_width > max_width:
            max_width = img_width
            max_image = filename
        if img_height > max_height:
            max_height = img_height
            max_image = filename

    return max_width, max_height, max_image

if __name__ == "__main__":
    # Replace 'input_dir' with the path to the directory containing the .jpg images
    input_dir = '/path/temp_directory'
    
    max_width, max_height, max_image = find_max_dimensions(input_dir)
    print(f"Maximum Width: {max_width}, Maximum Height: {max_height}")
    print(f"Image with Maximum Dimensions: {max_image}")

# +
import os
import shutil

def collect_and_save_jpg_files(input_dir, output_dir):
    """
    Recursively collects all .jpg files from input_dir and saves them into output_dir.
    
    Args:
    input_dir (str): The input directory to search for .jpg files.
    output_dir (str): The output directory where all .jpg files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk through the input directory and its subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Get the full path of the source file
                source_path = os.path.join(root, file)

                # Get the destination path in the output directory
                destination_path = os.path.join(output_dir, file)

                # Copy the .jpg file to the output directory
                shutil.copyfile(source_path, destination_path)

if __name__ == "__main__":
    # Replace 'input_dir' with the path to the source directory containing .jpg files
    # Replace 'output_dir' with the path to the directory where you want to save all .jpg files
    input_dir = '/path/Clean Dataset'
    output_dir = '/path/all_images'

    collect_and_save_jpg_files(input_dir, output_dir)

# +
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

if __name__ == "__main__":
    # Replace 'input_dir' with the path to the directory containing the .jpg images
    # Replace 'output_dir' with the path to the directory where you want to save the stacked images
    input_dir = '/path/all_images'
    output_dir = '/path/stacked_images'
    
    stack_similar_images(input_dir, output_dir)


# +
def get_npy(df, image_path):
    npy_files = []
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))

    # Extract barcode from file names and match with 'Barcode' column in the DataFrame
    barcode_from_file = [os.path.basename(file).replace('.npy', '').split('_')[1] for file in npy_files]
    matched_files = [file for file in npy_files if os.path.basename(file).replace('.npy', '').split('_')[1] in df['Barcode'].values]

    # Filter files based on matched barcodes
    filtered_files = [file for file in matched_files if os.path.basename(file).replace('.npy', '').split('_')[1] in barcode_from_file]

    return filtered_files


# +
df = pd.read_csv('/path/Train_Val_Holdout.csv') #Modify to contain path of .csv file with all samples from train/val/holdout.

def get_image_files_from_df(path):
    global df
    # Create a dictionary to store file paths indexed by replicate names
    file_paths = {}

    # Iterate through the directory and populate the dictionary with file paths
    for file_path in path.iterdir():
        if file_path.is_file():
            file_paths[file_path.name] = file_path

    # Sort the file paths based on the order of replicate names in the DataFrame
    sorted_file_paths = []
    for replicate_name in df["Replicate"]:
        for file_name, path in file_paths.items():
            if replicate_name in file_name:
                sorted_file_paths.append(path)
    
    # Now sorted_file_paths contains the file paths ordered based on the replicate names
    return sorted_file_paths


# +
def create_file_to_barcode_mapping(df):
    # Create a dictionary mapping file names to barcode values
    file_to_barcode = {}
    for index, row in df.iterrows():
        barcode = row['Barcode']
        file_name = f"id_{barcode}.jpg"  # Modify this format based on your actual file names
        file_to_barcode[file_name] = barcode
    return file_to_barcode


def get_y_class(image_path, df):
    # Extract the barcode from the image_path (assuming the barcode is part of the filename)
    barcode_from_path = Path(image_path).stem
    
    # Find the corresponding row in the DataFrame where the "Barcode" column matches the barcode from the image
    matching_row = df[df['Barcode'].apply(lambda x: x in barcode_from_path)]

    # If multiple matches are found, you may need to handle it accordingly (e.g., take the first match)
    if len(matching_row) > 1:
        matching_row = matching_row.iloc[0]

    # Extract the target value (Yield) from the matching row
    target_value = matching_row['Yield'].values[0]

    # Determine the class label based on yield ranges
    high_yield_range = 10
    medium_low_boundary = 6
    medium_high_boundary = 9.99

    if target_value > high_yield_range:
        class_label = "High Yield"
    elif medium_low_boundary <= target_value <= medium_high_boundary:
        class_label = "Medium Yield"
    else:
        class_label = "Low Yield"

    return class_label


def get_y(image_path):
    global df
    # Extract the barcode from the image_path (assuming the barcode is part of the filename)
    barcode_from_path = Path(image_path).stem
    
    # Find the corresponding row in the DataFrame where the "Barcode" column matches the barcode from the image
    matching_row = df[df['Replicate'].apply(lambda x: x in barcode_from_path)]
    
    # If multiple matches are found, you may need to handle it accordingly (e.g., take the first match)
    if len(matching_row) > 1:
        matching_row = matching_row.iloc[0]

    # Extract the target value (Yield) from the matching row and convert it to a scalar value
    target_value = matching_row['Yield']
    
    return target_value.item()


def mix_npy_blocks(img):
    "This function will be used to build the plot image and add transforms"
    # Cut the image in half and stack the chunks side-by-side
    chunk0 = img[:40, :20, :]
    chunk1 = img[40:80, :20, :]  

    if random.choice([True,False]):
        chunk0 = np.flip(chunk0[:,:,:], axis=0) # Flip vertically equals img[X,:,:]
    if random.choice([True,False]):
        chunk1 = np.flip(chunk1[:,:,:], axis=0) # Flip vertically equals img[X,:,:]
    if random.choice([True,False]):
        chunk0 = np.flip(chunk0[:,:,:], axis=1) # Flip horizontally equals img[:,X,:]
    if random.choice([True,False]):
        chunk1 = np.flip(chunk1[:,:,:], axis=1) # Flip horizontally equals img[:,X,:]

    if random.choice([True,False]):
        new_img = np.hstack((chunk0, chunk1))
    else:
        new_img =np.hstack((chunk1, chunk0))
    
    return  new_img

def vegetation_idxs(img):
    "Calculate VI and add as new bands"
    e = 0.00015 # Add a small value to avoid division by zero
    im = img
    
    # Calculate the VIs - change to np functions
    ndvi = np.divide(np.subtract(im[:,:,4], im[:,:,2]), (np.add(im[:,:,4], im[:,:,2])+e))
    ndvi_re = (im[:,:,4] - im[:,:,3]) / ((im[:,:,4] + im[:,:,3]) + e)
    ndre = (im[:,:,3] - im[:,:,2]) / ((im[:,:,3] + im[:,:,3]) + e) 
    envi = ((im[:,:,4] + im[:,:,1]) - (2 * im[:,:,0])) / (((im[:,:,4] - im[:,:,1]) + (2 * im[:,:,0])) + e)
    ccci = ndvi_re / (ndvi + e)
    gndvi = (im[:,:,4] - im[:,:,1])/ ((im[:,:,4] + im[:,:,1]) + e)
    gli = ((2* im[:,:,1]) - im[:,:,0] - im[:,:,2]) / (((2* im[:,:,1]) + im[:,:,0] + im[:,:,2]) + e)
    osavi = ((im[:,:,4] - im[:,:,3])/ ((im[:,:,4] + im[:,:,3] + 0.16)) *(1 + 0.16) + e)
    
    vi_list = [ndvi, ndvi_re, ndre, envi, ccci, gndvi , gli, osavi]
    vis = np.zeros((40,40,13)) 
    
    vis_stacked = np.stack(vi_list, axis=2)
    vis[:,:,:5] = im
    vis[:,:,5:] = vis_stacked
    
    return vis

def load_npy(fn):
    im = np.load(str(fn), allow_pickle=True)
    im = im*3 # increase image signal
    
    # Padding with zeros
    w, h , c = im.shape
    im = np.pad(im, ((0, 100-w), (0, 100-h), (0,0)),mode='constant', constant_values=0)    
    im = mix_npy_blocks(im) # Add transforms and stacking
    im = vegetation_idxs(im) # Add vegetation indexes bands
    # Normalise bands by deleting no-data values
    for band in range(13):
        im[:,:,band] = np.clip(im[:,:,band], 0, 1)
    
    # Swap axes because np is:  width, height, channels
    # and torch wants        :  channel, width , height
    im = np.swapaxes(im, 2, 0)
    im = np.swapaxes(im, 1, 2) 
    im = np.nan_to_num(im)
    return torch.from_numpy(im)

class MSITensorImage(TensorImage):
    _show_args = {'cmap':'Rdb'}
    
    def show(self, channels=3, ctx=None, vmin=None, vmax=None, **kwargs):
        "Visualise the images"
        if channels == 3 :
            return show_composite(self, 3, ctx=ctx, **{**self._show_args, **kwargs}) 
    
        else:
            return show_single_channel(self, channels, ctx=ctx, **{**self._show_args, **kwargs} )
    
    @classmethod
    def create(cls, fn:(Path, str), **kwargs) -> None:
        " Uses the load fn the array and turn into tensor"
        return cls(load_npy(fn))
        
    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'

def MSITensorBlock(cls=MSITensorImage):
    " A `TransformBlock` for numpy array images"
    # Calls the class create function to transform the x input using custom functions
    return TransformBlock(type_tfms=cls.create, batch_tfms=None)

def root_mean_squared_error(p, y): 
    return torch.sqrt(F.mse_loss(p.view(-1), y.view(-1)))

def create_rgb(img):
    # make RGB plot to visualise the "show batch"
    RGB = np.zeros((3, 40, 40))
    RGB[0] = img[2]
    RGB[2] = img[0]
    RGB[1] = img[1]
    #Change from tensor format to pyplot
    RGB = np.swapaxes(RGB, 0, 2)
    RGB = np.swapaxes(RGB, 1, 0)
    RGB = RGB 
    return RGB

def show_composite(img, channels, ax=None,figsize=(3,3), title=None, scale=True,
                   ctx=None, vmin=0, vmax=1, scale_axis=(0,1), **kwargs)->plt.Axes:
    "Show three channel composite"
    ax = ifnone(ax, ctx)
    dims = img.shape[0]
    RGBim = create_rgb(img)
    ax.imshow(RGBim)
    ax.axis('off')
    if title is not None: ax.set_title(title)
    return ax

def show_single_channel(img, channel, ax=None, figsize=(3,3), ctx=None, 
                        title=None, **kwargs) -> plt.Axes:
    ax = ifnone(ax, ctx)
    if ax is None: _, ax = plt.subplots(figsize=figsize)    
    
    tempim = img.data.cpu().numpy()
    
    if tempim.ndim >2:
        ax.imshow(tempim[channel,:,:])
        ax.axis('off')
        if title is not None: ax.set_title(f'{fname} with {title}')
    else:
        ax.imshow(tempim)
        ax.axis('off')
        if title is not None: ax.set_title(f'{fname} with {title}')
        
    return ax
