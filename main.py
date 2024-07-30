import torch
from torch.utils.data import DataLoader, Subset
import time
import os

from dataloader import CelebADataset
from frequency_network import Luna_Net
from discriminator import Discriminator
from loss import CombinedLoss
#hyperparameters
path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(path, "dataset/CelebA-HQ")
mask_path = os.path.join(path, "dataset/nvidia_irregular_masks_cleaned")
os.makedirs("results", exist_ok=True)
log_file = os.path.join(path, "results/training_log.txt")
batch_size = 32
learning_rate = 5 * 10e-5 #try another, 2x105 ok, 2x104 really bad 
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

#load the checkpoints if exist
checkpoint_path = os.path.join(path, "results/cloud_latest.pth")
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
    epoch_start_time = time.time()
    cumulative_time = 0
    epoch_loss = 0
    gen.train()
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

    #losses logging 
    epoch_end_time = time.time()
    epoch_time = (epoch_end_time - epoch_start_time) / 60

    #save the state of the model every epoch
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
        #save the image examples for the trained model every epoch
        with torch.no_grad():
            for i, (image, mask, target) in enumerate(zip(test_images, test_masks, test_targets)):
                if i >= 10:
                    break
                _, inpainted_img = gen(
                    image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
                )
                inpainted_img = inpainted_img.squeeze(0).cpu().detach()
        
    # Log the results
    with open(log_file, 'a') as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_time:.2f} mins\n")

print("Training completed.")