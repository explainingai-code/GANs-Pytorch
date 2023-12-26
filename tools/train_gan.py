import torch
import os
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations used for creating
# and training GAN
LATENT_DIM = 64
# For colored mnist change below to 3
IM_CHANNELS = 1
IM_PATH = 'data/train/images'
IM_EXT = 'png'
IM_SIZE = (28, 28)
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_SAMPLES = 225
NROWS = 15
##################


class Generator(nn.Module):
    r"""
    Generator for this gan is list of layers where each layer has the following:
    1. Linear Layer
    2. BatchNorm
    3. Activation(Tanh for last layer else LeakyRELU)
    The linear layers progressively increase dimension
    from LATENT_DIM to IMG_H*IMG_W*IMG_CHANNELS
    """
    def __init__(self):
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.img_size = IM_SIZE
        self.channels = IM_CHANNELS
        activation = nn.LeakyReLU()
        layers_dim = [self.latent_dim, 128, 256, 512, self.img_size[0] * self.img_size[1] * self.channels]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.BatchNorm1d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.channels, self.img_size[0], self.img_size[1])
        return out


class Discriminator(nn.Module):
    r"""
    Discriminator mimicks the design of generator
    only reduces dimensions progressive rather than increasing.
    From IMG_H*IMG_W*IMG_CHANNELS it reduces all the way to 1 where
    the last value is the probability discriminator thinks that
    given image is real(closer to 1 if real else closer to 0)
    """
    def __init__(self):
        super().__init__()
        self.img_size = IM_SIZE
        self.channels = IM_CHANNELS
        activation = nn.LeakyReLU()
        layers_dim = [self.img_size[0] * self.img_size[1] * self.channels, 512, 256, 128, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.LayerNorm(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x.reshape(-1, self.img_size[0] * self.img_size[1] * self.channels)
        for layer in self.layers:
            out = layer(out)
        return out
    

def train():
    # Create the dataset
    mnist = MnistDataset('train', im_path=IM_PATH, im_ext=IM_EXT)
    mnist_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
    
    # Instantiate the model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.train()
    discriminator.train()
    
    # Specify training parameters
    optimizer_generator = Adam(generator.parameters(), lr=1E-4, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=1E-4, betas=(0.5, 0.999))
    
    # Criterion is bcewithlogits hence no sigmoid in discriminator
    criterion = torch.nn.BCEWithLogitsLoss()

    # Run training
    steps = 0
    generated_sample_count = 0
    for epoch_idx in range(NUM_EPOCHS):
        generator_losses = []
        discriminator_losses = []
        mean_real_dis_preds = []
        mean_fake_dis_preds = []
        for im in tqdm(mnist_loader):
            real_ims = im.float().to(device)
            batch_size = real_ims.shape[0]
            
            # Optimize Discriminator
            optimizer_discriminator.zero_grad()
            fake_im_noise = torch.randn((batch_size, LATENT_DIM), device=device)
            fake_ims = generator(fake_im_noise)
            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)
            
            disc_real_pred = discriminator(real_ims)
            disc_fake_pred = discriminator(fake_ims.detach())
            disc_real_loss = criterion(disc_real_pred.reshape(-1), real_label.reshape(-1))
            mean_real_dis_preds.append(torch.nn.Sigmoid()(disc_real_pred).mean().item())

            disc_fake_loss = criterion(disc_fake_pred.reshape(-1), fake_label.reshape(-1))
            mean_fake_dis_preds.append(torch.nn.Sigmoid()(disc_fake_pred).mean().item())
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            optimizer_discriminator.step()
            ########################
            
            # Optimize Generator
            optimizer_generator.zero_grad()
            fake_im_noise = torch.randn((batch_size, LATENT_DIM), device=device)
            fake_ims = generator(fake_im_noise)
            disc_fake_pred = discriminator(fake_ims)
            gen_fake_loss = criterion(disc_fake_pred.reshape(-1), real_label.reshape(-1))
            gen_fake_loss.backward()
            optimizer_generator.step()
            ########################
            
            generator_losses.append(gen_fake_loss.item())
            discriminator_losses.append(disc_loss.item())
            
            # Save samples
            if steps % 50 == 0:
                with torch.no_grad():
                    generator.eval()
                    infer(generated_sample_count, generator)
                    generated_sample_count += 1
                    generator.train()
            #############
            steps += 1
        print('Finished epoch:{} | Generator Loss : {:.4f} | Discriminator Loss : {:.4f} | '
              'Discriminator real pred : {:.4f} | Discriminator fake pred : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(generator_losses),
            np.mean(discriminator_losses),
            np.mean(mean_real_dis_preds),
            np.mean(mean_fake_dis_preds),
        ))
        torch.save(generator.state_dict(), 'generator_ckpt.pth')
        torch.save(discriminator.state_dict(), 'discriminator_ckpt.pth')
    
    print('Done Training ...')


def infer(generated_sample_count, generator):
    r"""
    Method to save the generated samples
    :param generated_sample_count: Filename to save the output with
    :param generator: Generator model with trained parameters
    :return:
    """
    fake_im_noise = torch.randn((NUM_SAMPLES, LATENT_DIM), device=device)
    fake_ims = generator(fake_im_noise)
    ims = torch.clamp(fake_ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=NROWS)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists('samples'):
        os.mkdir('samples')
    img.save('samples/{}.png'.format(generated_sample_count))


if __name__ == '__main__':
    train()
