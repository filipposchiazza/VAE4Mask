import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import checkpoint, swa_update, dice_coefficient


class VaeTrainer():

    def __init__(self,
                 model,
                 optimizer,
                 device,
                 swa_model=None):
        """Variational Autoencoder Trainer.

        Parameters:
        -----------
        model : nn.Module
            VAE model.
        optimizer : torch optimizer
            Optimizer.
        device : torch device
            Device to use.
        swa_model : nn.Module
            Stochastic Weight Averaging model. Default is None
        """
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        if swa_model is not None:
            self.swa_model = swa_model.to(device)


    def train(self,
              train_dataloader,
              num_epochs,
              kl_weight,
              save_folder,
              val_dataloader=None,
              grad_clip=None,
              checkpoint_every=5):
        """Train the VAE.

        Parameters:
        -----------
        train_dataloader : torch DataLoader
            Training data loader.
        num_epochs : int
            Number of epochs.
        kl_weight : float
            KL weight.
        save_folder : str
            Directory to save model and generated samples checkpoints. 
        val_dataloader : torch DataLoader
            Validation data loader. Default is None.
        grad_clip : float
            Gradient clipping. Default is None.
        checkpoint_every : int
            Checkpoint every n epochs. Default is 5.
        """
        
        history = {'train_total_loss': [],
                   'train_reconstruction_loss': [],
                   'train_kl_divergence': [],
                   'train_dice_coefficient': [],
                   'val_total_loss': [],
                   'val_reconstruction_loss': [],
                   'val_kl_divergence': [],
                   'val_dice_coefficient': []}
        
        for epoch in range(num_epochs):

            # Training mode
            self.model.train()

            # Train one epoch
            train_losses = self._train_one_epoch(train_dataloader=train_dataloader,
                                                 kl_weight=kl_weight,
                                                 epoch=epoch,
                                                 grad_clip=grad_clip)

            # Update history
            history['train_total_loss'].append(train_losses[0])
            history['train_reconstruction_loss'].append(train_losses[1])
            history['train_kl_divergence'].append(train_losses[2])
            history['train_dice_coefficient'].append(train_losses[3])

            if val_dataloader is not None:
                # Validation mode
                self.model.eval()

                # Evaluate the model
                val_losses = self._validate(val_dataloader,
                                            kl_weight=kl_weight)

                # Update history
                history['val_total_loss'].append(val_losses[0])
                history['val_reconstruction_loss'].append(val_losses[1])
                history['val_kl_divergence'].append(val_losses[2])
                history['val_dice_coefficient'].append(val_losses[3])

            # Checkpoint
            checkpoint(model=self.model,
                       save_folder=save_folder,
                       current_epoch=epoch,
                       epoch_step=checkpoint_every,
                       batch_size=16)
            
            # SWA update
            if hasattr(self, 'swa_model'):
                swa_update(swa_model=self.swa_model,
                           model=self.model,
                           epoch=epoch,
                           epoch_step=checkpoint_every,
                           verbose=True)
        return history
    

    def _train_one_epoch(self,
                         train_dataloader,
                         kl_weight,
                         epoch,
                         grad_clip):
        
        running_total_loss = 0.0
        running_rec_loss = 0.0
        running_kl_loss = 0.0
        running_dice = 0.0

        mean_total_loss = 0.0
        mean_rec_loss = 0.0
        mean_kl_loss = 0.0
        mean_dice = 0.0

        self.optimizer.zero_grad()

        with tqdm(train_dataloader, unit='batches') as tepoch:

            for batch_idx, (masks, _) in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Load images to device
                masks = masks.to(self.device)

                # Forward pass
                x_pred, mean, log_var = self.model(masks)

                # Compute the losses
                rec_loss = F.binary_cross_entropy_with_logits(x_pred, masks, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                total_loss = rec_loss + kl_weight * kl_loss
                with torch.no_grad():
                    binary_pred = (x_pred > 0.5).float()
                    dice = dice_coefficient(binary_pred, masks)

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # Update the model parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update the running losses and mean losses
                running_total_loss += total_loss.item()
                running_rec_loss += rec_loss.item()
                running_kl_loss += kl_loss.item()
                running_dice += dice.item()

                mean_total_loss = running_total_loss / (batch_idx + 1)
                mean_rec_loss = running_rec_loss / (batch_idx + 1)
                mean_kl_loss = running_kl_loss / (batch_idx + 1)
                mean_dice = running_dice / (batch_idx + 1)

                # Update the progress bar
                tepoch.set_postfix(total_loss="{:.6f}".format(mean_total_loss),
                                   rec_loss="{:.6f}".format(mean_rec_loss),
                                   kl_loss="{:.6f}".format(mean_kl_loss),
                                   dice="{:.2f}".format(mean_dice))
        
        return mean_total_loss, mean_rec_loss, mean_kl_loss, mean_dice
    

    def _validate(self, 
                  val_dataloader,
                  kl_weight):
        
        running_total_loss = 0.0
        running_rec_loss = 0.0
        running_kl_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():
            for (masks, _) in val_dataloader:

                # Load images to device
                masks = masks.to(self.device)

                # Forward pass
                x_pred, mean, log_var = self.model(masks)

                # Compute the losses
                rec_loss = F.binary_cross_entropy_with_logits(x_pred, masks, reduction='mean')
                kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                total_loss = rec_loss + kl_weight * kl_loss
                binary_pred = (x_pred > 0.5).float()
                dice = dice_coefficient(binary_pred, masks)

                # Update the running losses
                running_total_loss += total_loss.item()
                running_rec_loss += rec_loss.item()
                running_kl_loss += kl_loss.item()
                running_dice += dice.item()

        mean_total_loss = running_total_loss / len(val_dataloader)
        mean_rec_loss = running_rec_loss / len(val_dataloader)
        mean_kl_loss = running_kl_loss / len(val_dataloader)
        mean_dice = running_dice / len(val_dataloader)

        print(f'Validation total loss: {mean_total_loss:.6f}, '
              f'Reconstruction loss: {mean_rec_loss:.6f}, '
              f'KL loss: {mean_kl_loss:.6f}',
              f'Dice Coefficient: {mean_dice:.2f}')
        
        return mean_total_loss, mean_rec_loss, mean_kl_loss, mean_dice
    