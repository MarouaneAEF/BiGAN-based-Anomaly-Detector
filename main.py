import torch 

from train import Trainer
from model import Discriminator, Generator
from dataloader import get_datasets


train_data, _  = get_datasets()


device = torch.device("cuda")



bigan = Trainer(train_data, Generator, Discriminator, device)

epochs = 200
for epoch in range(1, epochs + 1):

    bigan.training(epoch)
    bigan.test(epoch)
    
