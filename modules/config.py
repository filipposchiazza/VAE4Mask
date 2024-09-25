import torch

# Device configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Save configuration
SAVE_FOLDER = 'save'

# Dataset configuration
IMG_DIRS = ['dir1', 'dir2', 'dir3']
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05
FRACTION = 1.0
TRANSFORM = None

# Model configuration
IN_CHANNELS = 1
OUT_CHANNELS = 1
BASE_CHANNELS = 16
CHANNEL_MULTIPLIERS = [1, 2, 2, 4, 8]
NUM_RES_BLOCKS = [1, 1, 2, 2, 3]
LATENT_DIM = 256
IMG_SIZE = 256

# Training configuration
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
KL_WEIGHT = 1.0
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 5
