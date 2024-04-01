import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Generator(nn.Module):
    def __init__(self, num_emotions=5, hidden_dim=128, output_dim=10):
        super(Generator, self).__init__()

        # Define the architecture
        self.fc1 = nn.Linear(num_emotions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)  # Linear activation for the output layer
        return x


class GANTrainer:
    def __init__(self, generator, discriminator, surrogate, device='cpu'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.surrogate = surrogate.to(device)
        self.device = device

        # Ensure the discriminator's parameters are frozen
        for param in self.discriminator.parameters():
            param.requires_grad = False

        # Optimizer for the generator
        self.optim_g = optim.Adam(self.generator.parameters(), lr=0.002)

        # Loss function
        self.criterion = nn.BCELoss()

    def train(self, dataloader, epochs=10):
        for epoch in range(epochs):
            for emotion_labels, real_params in dataloader:
                # Convert emotion labels to one-hot encoding if necessary
                # and move tensors to the correct device
                emotion_labels = emotion_labels.to(self.device)
                real_params = real_params.to(self.device)

                # Generate fake parameters from the generator
                fake_params = self.generator(emotion_labels)

                # Use the surrogate model to process fake parameters
                fake_images = self.surrogate(fake_params)

                # Discriminator evaluation of fake images
                # Since discriminator's weights are fixed, we detach fake_images
                # to prevent gradients from flowing into the surrogate
                pred_fake = self.discriminator(fake_images.detach())
                loss_g = self.criterion(pred_fake, torch.ones_like(pred_fake))

                # Update generator
                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()

                # Logging, validation, or any additional steps would go here

            print(f"Epoch {epoch + 1}/{epochs}, Loss_G: {loss_g.item()}")
