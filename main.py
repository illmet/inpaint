import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from dataloader import CelebADataset
from frequency_network import Luna_Net
from discriminator import Discriminator
from loss import CombinedLoss

#hyperparameters
os.chdir("/users/pgrad/meti/Downloads")
path = os.getcwd()
os.makedirs("results/seventh_frequency_loss", exist_ok=True)
os.makedirs("results/seventh_frequency_results", exist_ok=True)
dataset_path = os.path.join(path, "dataset/CelebA-HQ")
mask_path = os.path.join(path, "dataset/nvidia_irregular_masks_cleaned")
batch_size = 16
learning_rate = 0.5 * 10e-4
num_epochs = 20
in_channels = 4
out_channels = 3
factor = 8

#data loading
dataset = CelebADataset(image_dir=dataset_path, mask_dir=mask_path)
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size
#indexing
indices = list(range(dataset_size))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_dataset, test_dataset = Subset(dataset, train_indices), Subset(
    dataset, test_indices
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_images, test_targets, test_masks = next(iter(test_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is currently set to {device}")

#load the models into the gpu
gen = Luna_Net(in_channels=in_channels, out_channels=out_channels, factor=factor)
gen.to(device)
disc = Discriminator()
disc.to(device)
total_gen = sum(p.numel() for p in gen.parameters())
total_disc = sum(p.numel() for p in disc.parameters())
print(f"Number of parameters in generator: {total_gen}")
print(f"Number of parameters in discriminator: {total_disc}")

#set up loss and optimizers
criterion = CombinedLoss().to(device)
optimizer_G = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(disc.parameters(), lr=learning_rate)

#set up the losses graphing
def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss over Epochs')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
train_losses = []
val_losses = []

#load the checkpoints if exist
checkpoint_path = "results/seventh_latest.pth"
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0

#training loop
for epoch in range(start_epoch, num_epochs):
    gen.train()
    cumulative_time = 0
    epoch_loss = 0
    for i, (images, targets, masks) in enumerate(train_loader):
        start_time = time.time()
        images, targets, masks = images.to(device), targets.to(device), masks.to(device)

        _, outputs = gen(images, masks)

        # Generator
        discriminator_output_on_generated = disc(outputs)

        generator_loss = criterion(
            outputs, targets, discriminator_output_on_generated, True
        )

        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

        # Discriminator
        discriminator_output_on_real = disc(targets)
        discriminator_real_loss = criterion.gan_loss(discriminator_output_on_real, True)

        # Discriminator loss for fake (generated) images
        discriminator_output_on_generated = disc(
            outputs.detach() 
        )  # detach to avoid backprop to generator
        discriminator_fake_loss = criterion.gan_loss(
            discriminator_output_on_generated, False
        )

        # Total discriminator loss
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        optimizer_D.zero_grad()
        discriminator_loss.backward()
        optimizer_D.step()

        #get the losses out for logging
        epoch_loss += generator_loss.item()

        #calculate the time
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        #print the rough logs 
        cumulative_time += elapsed_time
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Gen Loss: {generator_loss.item():.4f}, Disc Loss: {discriminator_loss.item():.4f}, Time Taken: {cumulative_time:.2f} mins"
            )
            cumulative_time = 0

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #save the state of the model every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(
            {
                "epoch": epoch,
                "gen_state_dict": gen.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
            },
            checkpoint_path,
        )

    #save the losses every epoch
    plot_losses(train_losses, val_losses, f"results/seventh_frequency_loss/loss_epoch_{epoch+1}.png")

    #Validation
    if (epoch + 1) % 1 == 0:
        gen.eval()
        val_loss = 0
        #validation loop to check the losses and append to the graphs
        with torch.no_grad():
            for images, targets, masks in test_loader:
                images, targets, masks = images.to(device), targets.to(device), masks.to(device)
                _, outputs = gen(images, masks)
                loss = criterion(outputs, targets, None, False)  # We don't need discriminator output for validation
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        #save the image examples for the trained model every epoch
        with torch.no_grad():
            for i, (image, mask, target) in enumerate(zip(test_images, test_masks, test_targets)):
                if i >= 10:
                    break
                _, inpainted_img = gen(
                    image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
                )
                inpainted_img = inpainted_img.squeeze(0).cpu().detach()
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
                plt.title("Corrupted Image")
                plt.subplot(1, 3, 2)
                plt.imshow(np.transpose(target.cpu().numpy(), (1, 2, 0)))
                plt.title("Ground Truth")
                plt.subplot(1, 3, 3)
                plt.imshow(np.transpose(inpainted_img.numpy(), (1, 2, 0)))
                plt.title("Inpainted Image")
                plt.savefig(f"results/seventh_frequency_results/inpainted_epoch_{epoch+1}_{i}.png")
                plt.close()