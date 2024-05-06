import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deepface import DeepFace
from models.neural_rendering import SurrogateVAE, Inference


class Generator(nn.Module):
    def __init__(self, num_emotions=7, hidden_dim=128, output_dim=33, num_params=77, base_path='', device="cuda" if torch.cuda.is_available() else
    "cpu"):
        super(Generator, self).__init__()
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
        # Compute the VAE Forward Pass
        vae_output = self.vae_evaluator.inference(rig_param_values)

        # Compute the DeepFace Forward Pass
        pred_dict = DeepFace.analyze(vae_output, actions=['emotion'], enforce_detection=False)[0]['emotion']
        pred_emotions = torch.tensor([pred_dict[key] for key in self.emotion_keys], dtype=torch.float32)

        # evtl. unnötig:
        pred_probabilities = F.softmax(pred_emotions, dim=0)

        target_emotions = torch.tensor(target_emotions, dtype=torch.float32)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred_emotions, target_emotions)
        return loss, pred_probabilities


class GANTrainer:
    def __init__(self, generator, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator.to(device)
        self.device = device

        # Optimizer for the generator
        self.optim_g = optim.Adam(self.generator.parameters(), lr=0.002)

    def train(self, dataloader, epochs=10):
        for epoch in range(epochs):
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

                # Logging, validation, or any additional steps would go here
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
