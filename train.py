from re import I
import torch 
from torch import device, optim 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.utils as vutils 

import numpy as np 

from model import Generator, Encoder, Discriminator
from utils.utils import weights_init_normal, weights_init



class Trainer:

    def __init__(self,  
                data,
                generator=Generator,
                encoder=Encoder,
                discriminator=Discriminator,  
                device="cuda",
                num_epochs=600, 
                lr_adam=1e-4, 
                lr_rmsprop=1e-4, 
                batch_size=64, 
                latent_dim=256, 
                wasserstein=False, 
                clamp=1e-2):


        self.generator = generator 
        self.encoder = encoder
        self.discriminator = discriminator
        self.num_epochs = num_epochs
        self.lr_adam = lr_adam
        self.lr_rmsprop = lr_rmsprop
        self.batch_sie = batch_size
        self.latent_dim = latent_dim
        self.wasserstein = wasserstein
        self.clamp= clamp
        self.train_loader = data
        self.device = device

    

    def train(self, epoch):

        G = self.generator(self.latent_dim).to(self.device)
        E = self.encoder(self.latent_dim).to(self.device)
        D = self.discriminator(self.latent_dim, self.wasserstein).to(self.device)

        G.train()
        E.train()
        D.train()

        G.apply(weights_init)
        E.apply(weights_init)
        D.apply(weights_init)

        if self.wasserstein:
            optimizer_ge = optim.RMSprop( list(G.parameters()) +
                                          list(E.parameters()), lr = self.lr_rmsprop)
            optimizer_d = optim.RMSprop(D.parameters(), lr=self.lr_rmsprop)
        else:
            optimizer_ge = optim.Adam(list(G.parameters()) +
                                      list(E.parameters()), lr=self.lr_adam)
            optimizer_d = optim.Adam(D.parameters(), lr=self.lr_adam)

        citerion = nn.BCELoss()

        i = 0
        ge_loss = 0
        d_loss = 0

        for x, _ in self.train_loader:
            # Defining labels 
            y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
            y_fake = Variable(torch.zeros(x.size(0),1)).to(self.device)

            # Noise for improving training 
            noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                        requires_grad=False).to(self.device)
            noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                        requires_grad=False).to(self.device)

            # Generator 
            z_fake = Variable(torch.randn((x.size(0), self.latent_dim, 1, 1)).to(self.device))#, requires_grad=False)
            x_fake = G(z_fake)

            # Encoder
            x_true = x.float().to(self.device)
            z_true = E(x_true)
            out_true = D(x_true + noise1, z_true)#.view(self.batch_sie, self.latent_dim, 1, 1))
            out_fake = D(x_fake + noise2, z_fake)
            
            # losses 
            if self.wasserstein:
                loss_d = -torch.mean(out_true) + torch.mean(out_fake)
                loss_ge = -torch.mean(out_fake) + torch.mean(out_true)
            else:
                loss_d = citerion(out_true.view(out_true.size(0),1), y_true) + citerion(out_fake.view(out_fake.size(0),1), y_fake)
                loss_ge = citerion(out_fake.view(out_fake.size(0),1), y_true) + citerion(out_true.view(out_true.size(0),1), y_fake)

            # Computing gradient and backpropagate.
            
            optimizer_ge.zero_grad()
            loss_ge.backward(retain_graph=True)
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_ge.step()
            optimizer_d.step()

            if self.wasserstein:
                for p in D.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)

            d_loss += loss_d.item()
            ge_loss += loss_ge.item()

            if i % 10 == 0:
                status = f"Epoch:{epoch}, Iter:{i}, D_Loss:{loss_d.item():>5f}, G_Loss:{loss_ge.item():>5f}, D(x):{out_true.mean().item():>5f}, D(G(x)):{out_fake.mean().item():>5f}"
                print(status)

            if i % 100 == 0: 
                vutils.save_image(x_fake.data[:10], f'./reconstruction/fake/fake_{epoch}_{i}.png')
                vutils.save_image(x_true.data[:10], f'./reconstruction/true/true_{epoch}_{i}.png')
                status = f"Training Epoch {epoch}, Avg Discriminator Loss: {(d_loss/len(self.train_loader)):>7f}, Avg Generator Loss: {(ge_loss/len(self.train_loader)):>7f}"
                print(status)
            i += 1

            if epoch % 50 == 0: 
                vutils.save_image(torch.cat([x_true.data[:16], x_fake.data[:16]]), f'./reconstruction/comparison/{epoch}.png')
                
                torch.save(E.state_dict(), f"./models/encoder_{epoch}.pth")
                torch.save(G.state_dict(), f"./models/generator_{epoch}.pth")
                torch.save(D.state_dict(), f"./models/discriminator_{epoch}.pth")

    def test(self, epoch):

        G = self.generator(self.latent_dim).to(self.device)
        E = self.encoder(self.latent_dim).to(self.device)
        D = self.discriminator(self.latent_dim, self.wasserstein).to(self.device)

        G.eval()
        E.eval()
        D.eval()

        with torch.no_grad():

            citerion = nn.BCELoss()

            
                
            ge_loss = 0
            d_loss = 0

            for x, _ in self.train_loader:
                # Defining labels 
                y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
                y_fake = Variable(torch.zeros(x.size(0),1)).to(self.device)

                # Noise for improving training 
                noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                            requires_grad=False).to(self.device)
                noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                            requires_grad=False).to(self.device)

                # Generator 
                z_fake = Variable(torch.randn((x.size(0), self.latent_dim, 1, 1)).to(self.device))#, requires_grad=False)
                x_fake = G(z_fake)

                # Encoder
                x_true = x.float().to(self.device)
                z_true = E(x_true)
                out_true = D(x_true + noise1, z_true)#.view(self.batch_sie, self.latent_dim, 1, 1))
                out_fake = D(x_fake + noise2, z_fake)
                
                # losses 
                if self.wasserstein:
                    loss_d = -torch.mean(out_true) + torch.mean(out_fake)
                    loss_ge = -torch.mean(out_fake) + torch.mean(out_true)
                else:
                    loss_d = citerion(out_true.view(out_true.size(0),1), y_true) + citerion(out_fake.view(out_fake.size(0),1), y_fake)
                    loss_ge = citerion(out_fake.view(out_fake.size(0),1), y_true) + citerion(out_true.view(out_true.size(0),1), y_fake)

                # Computing gradient and backpropagate.

                d_loss += loss_d.item()
                ge_loss += loss_ge.item()

            if epoch % 10 == 0:
                status = f"Epoch:{epoch}/{self.num_epochs}, Total D_Loss:{loss_d.item():>5f}, total G_Loss:{loss_ge.item():>5f}, D(x):{out_true.mean().item():>5f}, D(G(x)):{out_fake.mean().item():>5f}"
                print(status)

                sub_status = f"Test Epoch {epoch}, Avg Discriminator Loss: {(d_loss/len(self.train_loader)):>7f}, Avg Generator Loss: {(ge_loss/len(self.train_loader)):>7f}"
                print(sub_status)

            if epoch % 30 == 0: 
                vutils.save_image(torch.cat([x_true.data[:16], x_fake.data[:16]]), f'./reconstruction/test/{epoch}.png')   




        

