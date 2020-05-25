import argparse
import json
import multiprocessing
import random
from pathlib import Path

import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from art import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import (Discriminator16, Discriminator64, Discriminator128,
                           Generator16, Generator64, Generator128)


def train(batch_size, num_epochs, lr, manualSeed, image_size, data_path):
    """
    Main processes responsible for training Protein Gan.

    Inputs (hyperparameters):

    - batch_size: Number of maps shown to the generator and discriminator per step.
    - num_epochs: Number of epochs to train the GAN for.
    - lr:         Learning rate.
    - manualSeed: Set random seed for reproducibility. Default is 666.
    - image_size: NxN dimensions of images in data_path.
    - data_path:  Target hdf5 file containing datasets 'train_16', 'train_64' and 'train_128'

    Outputs:

    - model:      pt file containing discriminator and generator states, as well as optimizer states.
    - logs:       Logs are saved as tensorboard logs in 'runs/'.
    """

    # Setup
    Path("models").mkdir(exist_ok=True)
    models = Path("models")
    model_file = models / f"{image_size}_{batch_size}_{num_epochs}_{lr}_{manualSeed}.pt"

    print(f"Loading dataset from {data_path}")
    # Loading dataset
    with h5py.File(data_path, "r") as data_file:
        x = data_file[f"train_{image_size}"][:]
    data_len = len(x)

    # Scale down values by 100
    x = (x * 1) / 100

    # setting device on GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        ngpu = 1

    # Set random seed for reproducibility
    manualSeed = 666
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Tensorboard for tracking progress
    writer = SummaryWriter()

    # Number of workers for dataloader
    workers = multiprocessing.cpu_count()

    # Batch size during training
    batch_size = batch_size

    image_size = image_size

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = image_size

    # Size of feature maps in discriminator
    ndf = image_size

    # Number of training epochs
    num_epochs = num_epochs

    # Learning rate for optimizers
    lr = lr

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = ngpu

    dataloader = torch.utils.data.DataLoader(
        x, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # Decide which device we want to run on
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Create the generator
    if ngf == 16:
        netG = Generator16(ngpu).to(device)

    elif ngf == 64:
        netG = Generator64(ngpu).to(device)

    elif ngf == 128:
        netG = Generator128(ngpu).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    if ndf == 16:
        netD = Discriminator16(ngpu).to(device)

    elif ndf == 64:
        netD = Discriminator64(ngpu).to(device)

    elif ndf == 128:
        netD = Discriminator128(ngpu).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress
    iters = 0
    print("Starting Training")
    # For each epoch
    for epoch in tqdm(range(num_epochs), desc=f"Training Epochs"):
        # For each batch in the dataloader
        for i, data in tqdm(
            enumerate(dataloader, 0), total=int(data_len / batch_size), desc="Steps"
        ):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            # Unsqueezed dim one to convert [128, 64, 64] to [128, 1, 64, 64] to conform to D architecture
            real_cpu = (data.unsqueeze(dim=1).type(torch.FloatTensor)).to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Make Symmetric
            sym_fake = (fake.clamp(min=0) + fake.clamp(min=0).permute(0, 1, 3, 2)) / 2
            # Classify all fake batch with D
            output = netD(sym_fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(sym_fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()

            # Log to Tensorboard
            writer.add_scalars(
                "Disriminator Loss vs Generator Loss",
                {"Discriminator Loss": errD.item(), "Generator Loss": errG.item()},
                iters,
            )
            writer.add_scalar("Disriminator Accuracy", D_x, iters)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % int((data_len / batch_size) * 0.05) == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                writer.add_image(
                    "Generator Output",
                    vutils.make_grid(
                        (fake.clamp(min=0) + fake.clamp(min=0).permute(0, 1, 3, 2)) / 2,
                        padding=2,
                        normalize=True,
                    ),
                    iters,
                )

            iters += 1

    # Save our model
    torch.save(
        {
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimizerG_state_dict": optimizerG.state_dict(),
            "optimizerD_state_dict": optimizerD.state_dict(),
        },
        model_file,
    )
    print(f"Training successful! saving to {model_file}")


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to config json file", type=Path, dest="config"
    )
    parser.add_argument(
        "--batch_size",
        help="Number of maps shown to the generator and discriminator per step.",
        type=int,
        dest="batch_size",
    )
    parser.add_argument(
        "--manual_seed",
        help="Set random seed for reproducibility. Default is 666.",
        type=int,
        dest="manualSeed",
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train the GAN for.",
        type=int,
        dest="num_epochs",
    )
    parser.add_argument("--lr", help="Learning rate.", type=float, dest="lr")
    parser.add_argument("--data", help="Dataset location.", type=Path, dest="data_path")
    parser.add_argument("--image_size", help="Image size.", type=int, dest="image_size")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_json = json.load(f)
        batch_size = config_json["batch_size"]
        num_epochs = config_json["num_epochs"]
        lr = config_json["lr"]
        manualSeed = config_json["manualSeed"]
        image_size = config_json["image_size"]
        data_path = config_json["data_path"]
    else:
        batch_size = args.batch_size
        num_epochs = args.num_epochs
        lr = args.lr
        manualSeed = args.manualSeed
        image_size = args.image_size
        data_path = args.data_path

    Art = text2art("PROTEIN GAN", "larry3d")
    print(Art)
    print(
        f"""
    Welcome to Protein GAN! Your loaded parameters are:
    batch_size:  {batch_size}
    num_epochs:  {num_epochs}
    lr:          {lr}
    manual_seed: {manualSeed}
    image_size:  {image_size}
    """
    )
    train(batch_size, num_epochs, lr, manualSeed, image_size, data_path)
