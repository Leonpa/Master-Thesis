import torch


def save_checkpoint(model, model_optimizer, epoch, model_type):
    """
    Saves the current state of the training process depending on the model type.
    """
    if model_type == 'discriminator':
        file_path = "checkpoints/discriminator/checkpoint.pth"
    elif model_type == 'generator':
        file_path = "checkpoints/generator/checkpoint.pth"
    elif model_type == 'neural_rendering':
        file_path = "checkpoints/neural_rendering/checkpoint.pth"
    else:
        raise ValueError("Invalid model type. Please specify either 'discriminator', 'generator' or 'neural_rendering'.")

    torch.save({
        'state_dict': model.state_dict(),
        'model_optimizer': model_optimizer.state_dict(),
        'epoch': epoch
    }, file_path)


def load_checkpoint(model_type, model, model_optimizer):
    """
    Loads the training state from a checkpoint.
    """
    if model_type == 'discriminator':
        file_path = "checkpoints/discriminator/checkpoint.pth"
    elif model_type == 'generator':
        file_path = "checkpoints/generator/checkpoint.pth"
    elif model_type == 'neural_rendering':
        file_path = "checkpoints/neural_rendering/checkpoint.pth"
    else:
        raise ValueError("Invalid model type. Please specify either 'discriminator', 'generator' or 'neural_rendering'.")

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_optimizer.load_state_dict(checkpoint['model_optimizer'])
    return checkpoint['epoch']


def get_device():
    """
    Returns the device available for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_training_progress(epoch, epochs, d_loss, g_loss):
    """
    Logs the progress of training.
    """
    print(f"Epoch [{epoch}/{epochs}] | Discriminator Loss: {d_loss:.4f} | Generator Loss: {g_loss:.4f}")