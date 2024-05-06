import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class Emotion(nn.Module):
    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    Sequential:
    """
    Consruct emotion model, download and load weights
    """

    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    def forward(self, x):
        pass
        return x

    def loss_function(self, x, y):
        pass
        return x, y

class EmotionTrainer:
    def __init__(self, model, data_loader, learning_rate, num_epochs):
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        pass
        return self.model
