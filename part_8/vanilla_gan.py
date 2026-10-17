import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

# 1. Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.0002
LATENT_DIM = 100  # Size of the random noise vector
IMAGE_DIM = 28 * 28 * 1  # MNIST images are 28x28 pixels

# Create directory to save generated samples during training
os.makedirs("gan_images", exist_ok=True)

# 2. Prepare MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
])

dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=BATCH_SIZE, shuffle=True
)

# 3. Define the Generator Network
# Takes random noise and maps it to the image space
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, IMAGE_DIM),
            nn.Tanh()  # Outputs pixel values between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# 4. Define the Discriminator Network
# Takes an image and classifies whether it is real or fake (binary output)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(IMAGE_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# Loss function and Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Discriminator = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# 5. Training Loop
print("Starting Training...")
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):

        current_batch_size = imgs.size(0)

        # Labels for real (1) and fake (0) data
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss on real images
        real_pred = discriminator(imgs)
        d_real_loss = adversarial_loss(real_pred, real_labels)

        # Loss on fake images
        z = torch.randn(current_batch_size, LATENT_DIM)
        fake_imgs = generator(z)
        fake_pred = discriminator(fake_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake_labels)

        # Total Discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # The generator wants the discriminator to think fake images are real (1)
        gen_pred = discriminator(fake_imgs)
        g_loss = adversarial_loss(gen_pred, real_labels)

        g_loss.backward()
        optimizer_G.step()

    # Print progress metrics at the end of every epoch
    print(f"[Epoch {epoch+1}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save sample generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample_z = torch.randn(25, LATENT_DIM)
            sample_imgs = generator(sample_z)
            save_image(sample_imgs, f"gan_images/epoch_{epoch+1}.png", nrow=5, normalize=True)

print("Training finished! Check the 'gan_images' folder for generated digit previews.")