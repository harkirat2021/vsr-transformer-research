import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import *

""" Split video data into batches of sequences """
def batchify(video, seq_len, batch_size):
    x = video[:-1]
    y = video[1:]

    # Shape into batches of batch size and sequence length
    x = x.reshape(-1, batch_size, seq_len, video.shape[3], video.shape[1], video.shape[2])
    y = y.reshape(-1, batch_size, seq_len, video.shape[3], video.shape[1], video.shape[2])

    # Need to make seq_len come before batch size
    x = np.swapaxes(x, 1, 2)
    y = np.swapaxes(y, 1, 2)

    return x, y

""" Run training for transformer """
def train_transformer(model, x, y, n_epochs):
    model.train()
    losses = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    src_mask = autoencoder.generate_square_subsequent_mask(video_data.shape[1]).to(device)

    # Run epochs
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        # Run batch optim steps
        for inputs, targets in zip(x, y):
            # Clear gradients
            optimizer.zero_grad()

            # Get model outputs
            outputs = autoencoder(inputs, src_mask)
            
            # Compute and backpropagate loss
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Update loss and num batches
            total_epoch_loss += loss.item()
            num_batches += 1

        # Update losses
        losses.append(total_epoch_loss / num_batches)
        print("Epoch {}: Loss {}".format(epoch, losses[-1]))

        return model, losses

""" Run training """
def run_train(use_gpu=True):
    print("Training...")

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Using {}".format(device))

    # Init model
    model = TransformerModel(c=3, h=36, w=64, embed_dim=8, n_head=4, h_dim=20, n_layers=2, dropout=0.5)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {}".format(num_params))

    # Init data
    # ...

    # Train model
    model, losses = train_transformer(model, x, y, n_epochs)

    print("Done")