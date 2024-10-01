import numpy as np
import cv2
import os
import random
import config
from vae import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt


def generate_synthetic_masks(num_cells, vae_model, size=256):
    """ Generate one synthetic patch using the VAE model

    Parameters:
    -----------
    num_cells : int
        The number of cells to generate in the patch.
    vae_model : VAE
        The trained VAE model.
    size : int
        The size of the patch.

    Returns:
    --------
    synthetic_mask : np.ndarray
        The synthetic mask.
    """
    # Generate single cell masks
    samples = vae_model.sample(num_cells, binary=True).cpu().numpy()

    # Create the synthetic mask
    synthetic_mask = np.zeros((size, size), dtype=np.uint8)

    # Crop the single cell masks
    cropped_masks = []
    for sample in samples:
        # check if sample is empty
        if np.sum(sample) < 10:
            continue
        cropped_mask = crop_object(sample[0])
        cropped_masks.append(cropped_mask)

    # Place the cropped masks in the synthetic mask randomly and without overlapping
    for mask in cropped_masks:
        placed = False
        counter = 0
        while not placed:
            y = np.random.randint(0, size - mask.shape[0])
            x = np.random.randint(0, size - mask.shape[1])
            if can_place(synthetic_mask, mask, (y, x)):
                synthetic_mask[y:y+mask.shape[0], x:x+mask.shape[1]] = mask
                placed = True
            counter += 1
            if counter > 100:
                break

    return synthetic_mask



def crop_object(binary_mask):
    """ Crop the object from the binary mask 
    
    Parameters:
    -----------
    binary_mask : np.ndarray
        The binary mask to crop.
    
    Returns:
    --------
    np.ndarray
        The cropped mask.
    """

    # Convert to uint8
    binary_mask = binary_mask.astype(np.uint8)  
    # Find contours
    #coords = np.column_stack(np.where(binary_mask == 1))
    contours = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    # Get the bounding box of the non-zero pixels
    x, y, w, h = cv2.boundingRect(contours)
    
    # Crop the image using the bounding box coordinates
    cropped_mask = binary_mask[y:y+h, x:x+w]
    
    return cropped_mask



def can_place(patch, mask, top_left):
    """ Check if the mask can be placed at the top_left position without overlap

    Parameters:
    -----------
    patch : np.ndarray
        The patch where the mask will be placed.
    mask : np.ndarray
        The cropped mask to be placed.
    top_left : tuple
        The top-left corner of the mask. The tuple contains the y and x coordinates.
    
    Returns:
    --------
    bool
        True if the mask can be placed, False otherwise.
    """
    y, x = top_left
    h, w = mask.shape
    if y + h > patch.shape[0] or x + w > patch.shape[1]:
        return False
    return np.all(patch[y:y+h, x:x+w] == 0)



def place_masks_randomly(cropped_masks, final_size=256):
    """ Place all masks randomly within the final patch 
    
    Parameters:
    -----------
    cropped_masks : list
        list of cropped masks.
    final_size : int
        The size of the final patch.
    
    Returns:
    --------
    final_patch : np.ndarray
        The final patch with all masks placed.
    """
    final_patch = np.zeros((final_size, final_size), dtype=np.uint8)
    
    for mask in cropped_masks:
        placed = False
        while not placed:
            # Generate random top-left corner
            x = random.randint(0, final_size - mask.shape[0])
            y = random.randint(0, final_size - mask.shape[1])
            if can_place(final_patch, mask, (x, y)):
                final_patch[x:x+mask.shape[0], y:y+mask.shape[1]] = mask
                placed = True
    return final_patch


def count_num_cells(mask):
    """ Count the number of objects in the binary mask

    Parameters:
    -----------
    mask : np.ndarray
        The binary mask.    
    
    Returns:
    --------
    num_objects : int
        The number of objects in the binary mask.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # The number of contours found is the number of objects
    num_objects = len(contours)
    
    return num_objects


def estimate_num_cells_distribution(filenames, num_samples):
    """ Estimate the distribution of the number of cells in the patches

    Parameters:
    -----------
    filenames : list
        List of filenames of the patches to estimate the distribution of the number of cells.
    num_samples : int
        The number of patches to use for the estimation.
    
    Returns:
    --------
    num_cells : list
        List of the number of cells in each patch.
    """
    num_cells = []
    for i in tqdm(range(num_samples)):
        mask = plt.imread(filenames[i])[:, 256:, 0].astype(np.uint8)
        num_objects = count_num_cells(mask)
        num_cells.append(num_objects)
    return num_cells





if __name__ == '__main__':

    # Load the trained VAE model
    model = VAE.load_model(save_folder=config.SAVE_FOLDER).to(config.DEVICE)

    # Estimate the distribution of the number of cells in each patch
    ctr_files = os.listdir(config.IMG_DIRS[0])
    mds_files = os.listdir(config.IMG_DIRS[1])
    aml_files = os.listdir(config.IMG_DIRS[2])

    ctr_filenames = [os.path.join(config.IMG_DIRS[0], file) for file in ctr_files]
    mds_filenames = [os.path.join(config.IMG_DIRS[1], file) for file in mds_files]
    aml_filenames = [os.path.join(config.IMG_DIRS[2], file) for file in aml_files]

    filenames = ctr_filenames + mds_filenames + aml_filenames
    np.random.shuffle(filenames)

    num_cells = estimate_num_cells_distribution(filenames, 10000)

    x, frequencies = np.unique(num_cells, return_counts=True)
    probabilities = frequencies / np.sum(frequencies)

    # Generate synthetic masks
    save_counter = len(os.listdir(config.SYNTHETIC_MASKS_DIR))
    for i in range(config.NUM_SAMPLES):
        n = np.random.choice(x, size=1, p=probabilities).item()
        syn = generate_synthetic_masks(num_cells=n, vae_model=model, size=256)
        filename = os.path.join(config.SYNTHETIC_MASKS_DIR, f'syn_mask_{save_counter}.png')
        plt.imsave(filename, syn, cmap='gray')
        save_counter += 1
  