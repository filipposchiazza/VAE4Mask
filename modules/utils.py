import os
import torch


def checkpoint(model, save_folder, current_epoch, epoch_step, batch_size=None):
    """Save the model and generate samples at each epoch step.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to save.
    save_folder : str
        Folder to save the model.
    current_epoch : int
        Current epoch.
    epoch_step : int
        Epoch step to save the model.
    batch_size : int
        Number of samples to generate. Default is None to avoid generating samples.
    """

    if current_epoch != 0 and current_epoch % epoch_step == 0:
        # Create the checkpoint directory
        checkpoint_dir = os.path.join(save_folder, 'checkpoints', f'epoch_{current_epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model
        model.save_model(checkpoint_dir)
        # Samples generation
        if batch_size is not None:
            gen_imgs = model.sample(batch_size=batch_size)
            gen_imgs_file = os.path.join(checkpoint_dir, 'gen_imgs.pt')
            torch.save(gen_imgs, gen_imgs_file)

    return True



def swa_update(swa_model, model, epoch, epoch_step, verbose=True):
    """Update the SWA model at each epoch step.

    Parameters:
    -----------
    swa_model : torch.nn.Module
        SWA model.
    model : torch.nn.Module
        Model to update.
    epoch : int
        Current epoch.
    epoch_step : int
        Epoch step to update the SWA model.
    verbose : bool
        Verbose. Default is True.
    """

    if epoch != 0 and epoch % epoch_step == 0:
        swa_model.update_parameters(model)
        if verbose:
            print(f'SWA model updated at epoch {epoch}')
    return True

