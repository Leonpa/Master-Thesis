import torch
import os
import json
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        # Freeze VGG parameters
        for parameter in self.vgg.parameters():
            parameter.requires_grad = False
        self.criterion = nn.MSELoss()

        # Use the device of the model parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, target):
        # Ensure input and target are on the correct device
        input = input.to(self.device)
        target = target.to(self.device)

        # Normalize input and target to match VGG training data
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, -1, 1, 1)
        input = (input - mean) / std
        target = (target - mean) / std

        input_features = self.vgg(input)
        target_features = self.vgg(target)

        loss = self.criterion(input_features, target_features)
        return loss


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, num_params):
        super(ChannelAttention, self).__init__()
        # A simple linear layer to transform the rig parameters into attention scores
        self.attention = nn.Sequential(
            nn.Linear(num_params, num_channels),
            nn.Sigmoid()  # Ensure scores are in [0, 1] range
        )

    def forward(self, x, rig_params):
        # Generate attention scores for each channel
        scores = self.attention(rig_params).unsqueeze(2).unsqueeze(3)  # Reshape to match spatial dimensions
        return x * scores  # Apply attention scores to feature map


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features, num_params):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_params, num_features * 2)
        self.linear.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.linear.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, rig_params):
        # Obtain scale and bias
        scale_bias = self.linear(rig_params)
        scale, bias = scale_bias.chunk(2, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        bias = bias.unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale * x + bias


class SurrogateCNN(nn.Module):
    def __init__(self, num_params):
        super(SurrogateCNN, self).__init__()
        # Define the number of rigging parameters:

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input: 3x512x512
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 16x256x256
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x64x64
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64 + num_params, 1024),  # Combine with params
            nn.ReLU(),
        )

        self.adinorm1 = AdaptiveInstanceNorm(1024, num_params)

        self.attention1 = ChannelAttention(64, num_params)  # Assuming 64 channels in the feature map

        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # Output: 512x2x2
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: 16x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Output: 8x128x128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),  # Output: 4x256x256
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),  # Output: 3x512x512
            nn.Tanh(),
        )

    def forward(self, idle_image, rig_params):
        features = self.conv_layers(idle_image)
        attended_features = self.attention1(features, rig_params)
        combined_input = torch.cat((attended_features.view(attended_features.size(0), -1), rig_params), dim=1)
        intermediate = self.fc_layers(combined_input)
        intermediate = intermediate.view(-1, 1024, 1, 1)
        # normalized_features = self.adinorm1(intermediate, rig_params)
        output = self.upsample_layers(intermediate)
        return output


class RenderDataset(torch.utils.data.Dataset):
    def __init__(self, rig_params_json_path, train_set_path, idle_img_path, transform=None):
        self.img_folder = train_set_path
        self.idle_img_path = idle_img_path  # Path to the idle image
        self.transform = transform
        with open(rig_params_json_path, 'r') as f:
            self.params = json.load(f)
        self.keys = list(self.params.keys())

        # Load the idle image and apply transformations (do it here to avoid reloading it every time)
        self.idle_image = Image.open(self.idle_img_path).convert('RGB')
        if self.transform:
            self.idle_image = self.transform(self.idle_image)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        key = self.keys[idx]
        param_values = []
        for part in self.params[key].values():
            param_values.extend(part['location'])
            param_values.extend(part['rotation'])
        params = torch.tensor(param_values, dtype=torch.float32)

        # Load and transform the specific perturbed image
        img_path = os.path.join(self.img_folder, f'{key}_rendis0001.png')
        perturbed_image = Image.open(img_path).convert('RGB')
        if self.transform:
            perturbed_image = self.transform(perturbed_image)

        # Return both the idle image and the perturbed image, along with the parameters
        return self.idle_image, perturbed_image, params


class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=32, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else
    "cpu", log_dir='./logs', step_size=10, gamma=0.1):

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.step_size = step_size
        self.gamma = gamma

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)  # Initialize scheduler

        self.perceptual_loss = VGGLoss().to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.loss_weight = 0.5  # Weight for combining MSE and perceptual loss

        # TensorBoard SummaryWriter initialization
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (idle_images, perturbed_images, params) in enumerate(self.train_loader):
            idle_images, perturbed_images, params = idle_images.to(self.device), perturbed_images.to(self.device), params.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(idle_images, params)
            mse_loss = self.mse_loss(outputs, perturbed_images)
            vgg_loss = self.perceptual_loss(outputs, perturbed_images)
            # Combine losses
            loss = mse_loss + self.loss_weight * vgg_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * idle_images.size(0)
            # Log training loss every batch
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)

        # Log average loss at the end of the epoch
        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.writer.add_scalar('Loss/train_epoch_average', epoch_loss, epoch)

        return epoch_loss  # Returning for potential further usage

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            # Step the scheduler after each epoch
            self.scheduler.step()

        # Close the SummaryWriter after training is finished
        self.writer.close()


class Evaluator:
    def __init__(self, model, rig_params_json_path, eval_set_folder, idle_img_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.rig_params_json_path = rig_params_json_path
        self.img_folder = eval_set_folder
        self.idle_img_path = idle_img_path
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.model.eval()  # Set the model to evaluation mode

    def load_parameters(self, index=0):
        # Load the rig parameters from the JSON file
        with open(self.rig_params_json_path, 'r') as f:
            params_dict = json.load(f)

        selected_key = list(params_dict.keys())[index]  # Adjust based on the index
        selected_params = params_dict[selected_key]
        param_values = []
        for part in selected_params.values():
            param_values.extend(part['location'])
            param_values.extend(part['rotation'])
        return torch.tensor([param_values], dtype=torch.float32).to(self.device)  # Add batch dimension

    def load_idle_image(self):
        # Load and preprocess the idle image
        idle_image = Image.open(self.idle_img_path).convert('RGB')
        return self.transform(idle_image).unsqueeze(0).to(self.device)  # Add batch dimension

    def evaluate(self, index=0):
        rig_params = self.load_parameters(index)
        idle_image = self.load_idle_image()
        with torch.no_grad():
            output = self.model(idle_image, rig_params).cpu().squeeze(0)  # Remove batch dimension

        # Processing output for visualization
        output_image = output.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        print(max(output_image.flatten()))
        output_image = np.clip(output_image, 0, 1)  # Ensure the image's values are between 0 and 1

        ###############################################################################################
        output_image = output_image * 1
        ###############################################################################################

        # Return both the transformed idle image and the output for comparison
        idle_img_np = idle_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        return idle_img_np, output_image

    def display_results(self, idle_image, output_image):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(idle_image)
        plt.title('Original Idle Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title('Output Image')
        plt.axis('off')

        plt.show()
