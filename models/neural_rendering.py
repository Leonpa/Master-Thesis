import torch
import os
import json
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn.functional as F

"""
class FiLMLayer(nn.Module):
    def __init__(self, num_features, num_params=28):
        super(FiLMLayer, self).__init__()
        self.scale_transform = nn.Linear(num_params, num_features)  # Adjusted to match the flattened size of params
        self.shift_transform = nn.Linear(num_params, num_features)

    def forward(self, x, params):
        params_flat = params.view(params.size(0), -1)
        scale = self.scale_transform(params_flat).unsqueeze(2).unsqueeze(3).expand_as(x)
        shift = self.shift_transform(params_flat).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale + shift
"""


class FiLMLayer(nn.Module):
    def __init__(self, num_features, num_params):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features
        self.scale_transform = nn.Linear(num_params, num_features)
        self.shift_transform = nn.Linear(num_params, num_features)

    def forward(self, x, params):
        gamma = self.scale_transform(params).view(-1, self.num_features, 1, 1)
        beta = self.shift_transform(params).view(-1, self.num_features, 1, 1)
        return gamma * x + beta


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, num_params, chann_att_scale=1.0):
        super(ChannelAttention, self).__init__()
        self.chann_att_scale = chann_att_scale
        # A simple linear layer to transform the rig parameters into attention scores
        self.attention = nn.Sequential(
            nn.Linear(num_params, num_channels),
            nn.Sigmoid()  # Ensure scores are in [0, 1] range
        )

    def forward(self, x, rig_params):
        # Generate attention scores for each channel
        scores = self.attention(rig_params).unsqueeze(2).unsqueeze(3)  # Reshape to match spatial dimensions
        return x * scores  # * self.chann_att_scale


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features, num_params, adain_scale_factor=1.0):
        super(AdaptiveInstanceNorm, self).__init__()
        self.scale_factor = adain_scale_factor
        self.num_features = num_features
        self.linear = nn.Linear(num_params, num_features * 2)
        self.linear.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.linear.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, rig_params):
        # Obtain scale and bias
        scale_bias = self.linear(rig_params)
        scale, bias = scale_bias.chunk(2, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)  ### * self.scale_factor
        bias = bias.unsqueeze(2).unsqueeze(3).expand_as(x)  ###*  self.scale_factor
        return scale * x + bias


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SurrogateVAE(nn.Module):
    def __init__(self, num_channels=3, num_params=28, latent_dim=256):
        super().__init__()
        # Image processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch, 128, 64, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch, 256, 32, 32]
            nn.ReLU()
        )

        self.film_layer = FiLMLayer(256, num_params)  # Apply FiLM after some convolutional layers

        # Fully connected layers for combining conv output with parameters
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32 + num_params, 1024),  # Adjust this line to include num_params
            nn.ReLU(),
            nn.Linear(1024, 2 * latent_dim)  # Continue as before
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (256, 32, 32)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, params):
        x = self.conv_layers(x)
        # x = self.film_layer(x, params)  # Apply FiLM with parameters to modulate the features
        x = x.view(x.size(0), -1)  # Ensure x is flattened correctly
        x = torch.cat((x, params), dim=1)  # Concatenate x and params
        x = self.fc_layers(x)
        mu, log_var = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    @staticmethod
    def loss_function(recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD


class RenderDataset(torch.utils.data.Dataset):
    def __init__(self, rig_params_json_path, train_set_path, idle_img_path, transform=None):
        self.img_folder = train_set_path
        self.idle_img_path = idle_img_path  # Path to the idle image
        self.transform = transform
        with open(rig_params_json_path, 'r') as f:
            self.params = json.load(f)
        self.keys = list(self.params.keys())

        # Pre-compute mean and std for normalization
        self.params_mean, self.params_std = self.compute_params_stats()

        # Load the idle image and apply transformations (do it here to avoid reloading it every time)
        self.idle_image = Image.open(self.idle_img_path).convert('RGB')
        if self.transform:
            self.idle_image = self.transform(self.idle_image)

    def compute_params_stats(self):
        all_params = []
        for key in self.keys:
            param_values = []
            for part in self.params[key].values():
                param_values.extend(part['location'])
                param_values.extend(part['rotation'])
            all_params.append(param_values)

        all_params_tensor = torch.tensor(all_params, dtype=torch.float32)
        params_mean = torch.mean(all_params_tensor, dim=0)
        params_std = torch.std(all_params_tensor, dim=0)
        return params_mean, params_std

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        key = self.keys[idx]
        param_values = []
        for part in self.params[key].values():
            param_values.extend(part['location'])
            param_values.extend(part['rotation'])
        params = torch.tensor(param_values, dtype=torch.float32)

        params = (params - self.params_mean) / (self.params_std + 1e-8)  # Adding epsilon to avoid division by zero

        img_path = os.path.join(self.img_folder, f'{key}_rendis0001.png')
        perturbed_image = Image.open(img_path).convert('RGB')
        if self.transform:
            perturbed_image = self.transform(perturbed_image)

        # Return both the idle image and the perturbed image, along with the parameters
        return self.idle_image, perturbed_image, params


class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=32, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else
    "cpu", log_dir='./logs', lr_step_size=10, gamma=0.1, vgg_loss_weight=0.8, landmark_loss_weight=0.2):

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.step_size = lr_step_size
        self.gamma = gamma

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.lr_scheduler = True if lr_step_size > 0 else False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)  # Initialize scheduler

        self.perceptual_loss = VGGLoss().to(device)
        self.mse_loss = torch.nn.MSELoss()
        self.simulated_landmark_loss = SimulatedLandmarkLoss().to(device)
        self.vgg_loss_weight = vgg_loss_weight  # Weight for combining MSE and perceptual loss
        self.landmark_loss_weight = landmark_loss_weight  # Weight for combining perceptual and landmark loss

        # TensorBoard SummaryWriter initialization
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self, epoch):

        """
        self.model.train()
        running_loss = 0.0
        for batch_idx, (idle_images, perturbed_images, params) in enumerate(self.train_loader):
            idle_images, perturbed_images, params = idle_images.to(self.device), perturbed_images.to(self.device), params.to(self.device)

            self.model.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(idle_images, params)
            mse_loss = self.mse_loss(outputs, perturbed_images)
            vgg_loss = self.perceptual_loss(outputs, perturbed_images)
            landmark_loss = self.simulated_landmark_loss(outputs, perturbed_images)

            print(f"Loss after VGG: {vgg_loss}, Loss after MSE: {mse_loss}, Loss after Landmark: {landmark_loss}")
            loss = mse_loss + self.vgg_loss_weight * vgg_loss + self.landmark_loss_weight * landmark_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * idle_images.size(0)
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
        """

        self.model.train()
        running_loss = 0.0
        for batch_idx, (idle_images, perturbed_images, params) in enumerate(self.train_loader):
            idle_images, perturbed_images, params = idle_images.to(self.device), perturbed_images.to(self.device), params.to(self.device)

            self.optimizer.zero_grad()
            recon_images, mu, log_var = self.model(idle_images, params)
            loss = self.model.loss_function(recon_images, perturbed_images, mu, log_var)
            # vgg_loss = self.perceptual_loss(recon_images, perturbed_images)
            # loss = loss + self.vgg_loss_weight * vgg_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * idle_images.size(0)
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + batch_idx)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            # Step the scheduler after each epoch
            if self.lr_scheduler:
                self.scheduler.step()

        # Close the SummaryWriter after training is finished
        self.writer.close()


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])  # Use up to the second last layer

    def forward(self, x):
        x = self.features(x)  # (B, C, H, W)
        return x


class SimulatedLandmarkLoss(nn.Module):
    def __init__(self):
        super(SimulatedLandmarkLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze the feature extractor

    def forward(self, generated_images, target_images):
        gen_features = self.feature_extractor(generated_images)
        target_features = self.feature_extractor(target_images)
        # Compute loss as the mean squared error of the features
        loss = F.mse_loss(gen_features, target_features)
        return loss


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

    """
    def evaluate(self, index=0):
        rig_params = self.load_parameters(index)
        idle_image = self.load_idle_image()

        ground_truth_image_path = os.path.join(self.img_folder, f'{index+1}_rendis0001.png')
        ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')
        ground_truth_image = self.transform(ground_truth_image).unsqueeze(0).to(self.device)
        ground_truth_image_np = ground_truth_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        ground_truth_image_np = np.clip(ground_truth_image_np, 0, 1)

        with torch.no_grad():
            output = self.model(idle_image, rig_params).cpu().squeeze(0)  # Remove batch dimension

        # Processing output for visualization
        output_image = output.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        print(max(output_image.flatten()))
        output_image = np.clip(output_image, 0, 1)  # Ensure the image's values are between 0 and 1

        # Return both the transformed idle image and the output for comparison
        idle_img_np = idle_image.cpu().squeeze(0).permute(1, 2, 0).numpy()

        return idle_img_np, ground_truth_image_np, output_image
    """

    def evaluate(self, index=0):
        rig_params = self.load_parameters(index)
        idle_image = self.load_idle_image()

        ground_truth_image_path = os.path.join(self.img_folder, f'{index + 1}_rendis0001.png')
        ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')
        ground_truth_image = self.transform(ground_truth_image).unsqueeze(0).to(self.device)
        ground_truth_image_np = ground_truth_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        ground_truth_image_np = np.clip(ground_truth_image_np, 0, 1)

        with torch.no_grad():
            reconstructed_image, _, _ = self.model(idle_image, rig_params)  # Unpack the tuple
            output = reconstructed_image.cpu().squeeze(0)  # Now correctly calling cpu on the tensor

        # Processing output for visualization
        output_image = output.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
        output_image = np.clip(output_image, 0, 1)  # Ensure the image's values are between 0 and 1

        # Return both the transformed idle image and the output for comparison
        idle_img_np = idle_image.cpu().squeeze(0).permute(1, 2, 0).numpy()

        return idle_img_np, ground_truth_image_np, output_image

    def display_results(self, idle_image, ground_truth_image, output_image):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(idle_image)
        plt.title('Original Idle Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth_image)
        plt.title('Ground Truth Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(output_image)
        plt.title('Output Image')
        plt.axis('off')
        plt.show()


class LatentSpaceVisualizer:
    def __init__(self, model):
        self.model = model
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.model.eval()  # Ensure the model is in evaluation mode

    def encode(self, data_loader):
        """Encode data using the model's forward method."""
        self.model.eval()
        with torch.no_grad():
            for idle_images, perturbed_images, params in data_loader:
                idle_images, perturbed_images, params = idle_images.to(self.device), perturbed_images.to(self.device), params.to(self.device)

                # Assuming you want to use both images, or just one; adjust as necessary:
                outputs, mu, log_var = self.model(perturbed_images, params)  # Example using perturbed_images
                return mu, log_var  # Return first batch only for simplicity

    def reparameterize(self, mu, log_var):
        """ Sample from the latent space using the reparameterization trick. """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reduce_dimensions(self, z):
        """ Use t-SNE to reduce dimensions for visualization. """
        z = z.cpu().numpy()
        z_embedded = TSNE(n_components=2).fit_transform(z)
        return z_embedded

    def plot_latent_space(self, z_embedded, labels=None):
        """Plot the reduced latent space."""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels if labels is not None else 'blue')
        plt.colorbar(scatter) if labels is not None else None
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Latent Space Visualization')
        plt.show()
