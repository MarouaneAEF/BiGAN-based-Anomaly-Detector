import torch 

from train import Trainer
from model import Discriminator, Encoder, Generator
from dataloader import get_cifar_10





device = torch.device("cuda")

dataloader = get_cifar_10(batch_size=64)

bigan = Trainer(dataloader, Generator, Encoder, Discriminator, device)

epochs = 200
for epoch in range(1, epochs + 1):

    bigan.train(epoch)
    bigan.test(epoch)
    
