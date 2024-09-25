import os
import torch
from torch import optim
import config
from dataset import prepare_MaskDataset
from vae import VAE
from trainer import VaeTrainer




train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_MaskDataset(img_dirs=config.IMG_DIRS,
                                                                                   batch_size=config.BATCH_SIZE,
                                                                                   validation_split=config.VALIDATION_SPLIT,
                                                                                   fraction=config.FRACTION,
                                                                                   transform=config.TRANSFORM)

# Model configuration
model = VAE(in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            channel_multipliers=config.CHANNEL_MULTIPLIERS,
            num_res_blocks=config.NUM_RES_BLOCKS,
            latent_dim=config.LATENT_DIM,
            img_size=config.IMG_SIZE,
            device=config.DEVICE)

# Create SWA model
swa_model = optim.swa_utils.AveragedModel(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Trainer
trainer = VaeTrainer(model=model,
                     optimizer=optimizer,
                     device=config.DEVICE,
                     swa_model=swa_model)

# Train
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config.NUM_EPOCHS,
                        kl_weight=config.KL_WEIGHT,
                        save_folder=config.SAVE_FOLDER,
                        val_dataloader=val_dataloader,
                        grad_clip=config.GRAD_CLIP,
                        checkpoint_every=config.CHECKPOINT_EVERY)

# Save model, history and SWA model
model.save_model(config.SAVE_FOLDER)
model.save_history(history, config.SAVE_FOLDER)
swa_model_file = os.path.join(config.SAVE_FOLDER, 'swa_model.pt')
torch.save(swa_model.state_dict(), swa_model_file)

