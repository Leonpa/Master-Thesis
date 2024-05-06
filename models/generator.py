import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deepface import DeepFace
from models.neural_rendering import SurrogateVAE, Inference


class Generator(nn.Module):
    def __init__(self, num_emotions=7, hidden_dim=128, output_dim=77, num_params=77, base_path='', device="cuda" if torch.cuda.is_available() else
    "cpu"):
        super().__init__()
        self.fc1 = nn.Linear(num_emotions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.base_path = base_path
        self.num_params = num_params
        self.device = device
        self.emotion_keys = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        def full_path(relative_path):
            return os.path.join(base_path, relative_path)

        self.rig_params_json_path = full_path('data/render_trainset/rendering_rig_params.json')
        self.train_set_path = full_path('data/render_trainset')
        self.idle_img_path = full_path('data/render_trainset/idle.png')

        self.model = SurrogateVAE(num_params=self.num_params)
        self.model.load_state_dict(torch.load('checkpoints/Neural_Rendering/model_weights_77.pth', map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.vae_evaluator = Inference(self.model, self.idle_img_path, device=self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)  # Linear activation for the output layer
        return x

    def process_and_compute_loss(self, rig_param_values, target_emotions):
        losses = []
        pred_probabilities_list = []

        # Loop through each item in the batch
        for i in range(rig_param_values.size(0)):
            single_rig_param = rig_param_values[i].unsqueeze(0)  # Add batch dimension
            single_target_emotion = target_emotions[i].unsqueeze(0)  # Add batch dimension

            # Compute the VAE Forward Pass for single item
            vae_output = self.vae_evaluator.inference(single_rig_param)

            # Compute the DeepFace Forward Pass
            pred_dict = DeepFace.analyze(vae_output, actions=['emotion'], enforce_detection=False)[0]['emotion']
            pred_emotions = torch.tensor([pred_dict[key] for key in self.emotion_keys], dtype=torch.float32)
            pred_emotions = pred_emotions.unsqueeze(0)  # Ensure it's batched

            # Compute loss for single item
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(pred_emotions, single_target_emotion)

            losses.append(loss)
            pred_probabilities_list.append(pred_emotions)

        # Aggregate the losses
        total_loss = torch.stack(losses).mean()  # Average loss across the batch
        pred_probabilities_batch = torch.cat(pred_probabilities_list, dim=0)  # Concatenate all predictions

        return total_loss, pred_probabilities_batch


class GANTrainer:
    def __init__(self, generator, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator.to(device)
        self.device = device

        # Optimizer for the generator
        self.optim_g = optim.Adam(self.generator.parameters(), lr=0.002)

    def train(self, dataloader, epochs=10):
        for epoch in range(epochs):
            self.generator.model.train()  # Ensure the model is in training mode
            for emotion_labels, real_params in dataloader:
                # Convert emotion labels to one-hot encoding if necessary
                # and move tensors to the correct device
                emotion_labels = emotion_labels.to(self.device)
                real_params = real_params.to(self.device)

                # Generate fake parameters from the generator
                fake_params = self.generator(emotion_labels)

                # Compute loss using the Generator's internal loss computation
                loss, _ = self.generator.process_and_compute_loss(fake_params, emotion_labels)

                # Update generator
                self.optim_g.zero_grad()
                loss.backward()
                self.optim_g.step()

                # Switch back to evaluation mode after the training step
            self.generator.model.eval()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
