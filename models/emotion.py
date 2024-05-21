import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class Emotion(nn.Module):
    def __init__(self):
        super(Emotion, self).__init__()

        # Define the architecture of the model
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.pool = nn.MaxPool2d(5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.avgpool2 = nn.AvgPool2d(3, stride=2)
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, len(labels))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avgpool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.avgpool2(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class EmotionTrainer:
    def __init__(self, model, data_loader, learning_rate, num_epochs):
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')
        return self.model
