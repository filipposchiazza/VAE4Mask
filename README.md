# Variational Autoencoder for Binary Masks Generation
This repository contains a Torch-based implementation of a Variational Autoencoder (VAE) specifically designed for the generation of binary masks. The VAE is a type of generative model that can learn a compressed representation of input data, and then generate new data that resembles the original input. In this case, the VAE has been tailored to work with binary masks in the field of citology. The main difference with a standard VAE is that the loss function is a binary cross-entropy loss and not the classic mean squared error loss.

## Repository Structure
The files are organized as follows:
- `train.py`: Contains an example script to train the VAE model.
- `trainer.py`: Contains the class definition for the VAE trainer, that is the what will handle the training process.
- `vae.py`: Contains the implementation of the VAE model.
- `dataset.py`: Contains the implementation of the dataset class.
- `building_modules.py`: Contains the implementation of the encoder and decoder modules.
- `utils.py`: Contains utility functions, such as checkpoint saving and metrics definition.
- `config.py`: Contains the configuration parameters for the VAE model and the training process.
- `gen.py`: Contains an example script to generate binary masks using the trained VAE model.


## How to Use
Import the necessary dependencies:
```python
import os
import torch
from torch import optim
import config
from dataset import prepare_MaskDataset
from vae import VAE
from trainer import VaeTrainer
```

Load the dataset:
```python
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_MaskDataset(img_dirs=config.IMG_DIRS,
                                                                                   batch_size=config.BATCH_SIZE,
                                                                                   validation_split=config.VALIDATION_SPLIT,
                                                                                   fraction=config.FRACTION,
                                                                                   transform=config.TRANSFORM)
```

Model configuration
```python
model = VAE(in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            channel_multipliers=config.CHANNEL_MULTIPLIERS,
            num_res_blocks=config.NUM_RES_BLOCKS,
            latent_dim=config.LATENT_DIM,
            img_size=config.IMG_SIZE,
            device=config.DEVICE)
```

Create the SWA model, the optimizer and the trainer:
```python
# SWA model
swa_model = optim.swa_utils.AveragedModel(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Trainer
trainer = VaeTrainer(model=model,
                     optimizer=optimizer,
                     device=config.DEVICE,
                     swa_model=swa_model)
```

Train the VAE model:
```python
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        kl_weight=config.KL_WEIGHT,
                        save_folder=config.SAVE_FOLDER,
                        val_dataloader=val_dataloader,
                        grad_clip=config.GRAD_CLIP,
                        checkpoint_every=config.CHECKPOINT_EVERY)
```

Finally, save the model, the training history and the SWA model
```python
model.save_model(config.SAVE_FOLDER)
model.save_history(history, config.SAVE_FOLDER)
swa_model_file = os.path.join(config.SAVE_FOLDER, 'swa_model.pt')
torch.save(swa_model.state_dict(), swa_model_file)
```

## Dependencies
* python == 3.12
* pytorch == 2.3.1 
* tqdm == 4.66.4


