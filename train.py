from re import I
import torch 
from torch import device, optim 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.utils as vutils 
import torch.nn.functional as F
import numpy as np 

from model import Generator, Discriminator
from utils.utils import weights_init_normal, weights_init


class Trainer:

    def __init__(self,  
                train_data,
                test_data,
                generator=Generator,
                discriminator=Discriminator,  
                device=torch.device("cuda"),
                num_epochs=600, 
                lr_adam=1e-4, 
                lr_rmsprop=1e-4, 
                batch_size=64, 
                latent_dim=256):


        self.generator = generator 
        self.discriminator = discriminator
        self.num_epochs = num_epochs
        self.lr_adam = lr_adam
        self.lr_rmsprop = lr_rmsprop
        self.batch_sie = batch_size
        self.latent_dim = latent_dim
        self.train_loader = train_data
        self.test_loader = test_data
        self.device = device

    

    def training(self, epoch):
        G = self.generator(self.latent_dim).to(self.device)
        D = self.discriminator(self.latent_dim).to(self.device)
      
        G.train()
        D.train()

        G.apply(weights_init_normal)
        D.apply(weights_init_normal)

        
        
        optimizer_g = optim.Adam(G.parameters(), lr=self.lr_adam)
        optimizer_d = optim.Adam(D.parameters(), lr=self.lr_adam)

        i = 0
        g_loss = 0
        d_loss = 0
    
        for x, _ in self.train_loader:
            #  D training 
            D.zero_grad()
            ### REAL ENCODING ###
            y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
            # Noise for improving training 
            noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                        requires_grad=False).to(self.device)
            # ENCODING
            x_true = x.float().to(self.device)
            z_true = G.encoding(x_true)
            z_true = z_true.view(self.batch_sie, self.latent_dim, 1, 1)
            out_true = D(x_true + noise1, z_true)
            # d real loss
            loss_d_real = F.binary_cross_entropy(out_true.view(out_true.size(0),1), y_true)
            loss_d_real.backward(retain_graph=True)

            ### FAKE DECODING ###
            y_fake = Variable(torch.zeros(x.size(0),1)).to(self.device)
            # Noise for improving training 
            noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1*(self.num_epochs - epoch)/self.num_epochs),
                        requires_grad=False).to(self.device)
            # DECODING 
            z_fake = Variable(torch.randn((x.size(0), self.latent_dim, 1, 1)).to(self.device))#, requires_grad=False)
            x_fake = G.decoding(z_fake)
            out_fake = D(x_fake + noise2, z_fake)
            # d fake losses 
            loss_d_fake = F.binary_cross_entropy(out_fake.view(out_fake.size(0),1), y_fake)
            loss_d_fake.backward(retain_graph=True)
            # Computing loss of D as sum over the fake and the real batches
            loss_d = .5 * ( loss_d_real + loss_d_fake )
            ### UPDATE D ###
            optimizer_d.step()
            
            # G training 
            # backpropagate over the updated D parameters and the G parameters 
            G.zero_grad()
            # updated losses after updating D 
            y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
            out_fake = D(x_fake + noise2, z_fake)
            loss_gdecoding = F.binary_cross_entropy(out_fake.view(out_fake.size(0),1), y_true) 
            # maximizing log(D(G(z))) : 
            loss_gdecoding.backward()
            # update reconstruction loss after updating D
            out_true = D(x_true + noise1, z_true)
            y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
            loss_gencoding =  F.binary_cross_entropy(out_true.view(out_true.size(0),1), y_true)
            # minimizing reconstruction loss 
            loss_gencoding.backward()
            # Computing loss of G as sum over the fake encoding loss and generation loss
            loss_g = .5 * ( loss_gdecoding + loss_gencoding )
            ### UPDATING G ###
            optimizer_g.step()

            d_loss += loss_d.item()
            g_loss += loss_g.item()

            if i % 10 == 0:
                status = f"Epoch:{epoch}, Iter:{i}, D_Loss:{loss_d.item():.5f}, G_Loss:{loss_g.item():.5f}, D(x):{out_true.mean().item():.5f}, D(G(x)):{out_fake.mean().item():.5f}"
                print(status)

            if i % 100 == 0: 
                vutils.save_image(x_fake.data[:10], f'./reconstruction/fake/fake_{epoch}_{i}.png')
                vutils.save_image(x_true.data[:10], f'./reconstruction/true/true_{epoch}_{i}.png')
                status = f"Training Epoch {epoch}, Avg Discriminator Loss: {(d_loss/len(self.train_loader)):.5f}, Avg Generator Loss: {(g_loss/len(self.train_loader)):.5f}"
                print(status)
            i += 1

            if epoch % 50 == 0: 
                vutils.save_image(torch.cat([x_true.data[:16], x_fake.data[:16]]), f'./reconstruction/comparison/{epoch}.png')
                
                torch.save(G.state_dict(), f"./models/generator_{epoch}.pth")
                torch.save(D.state_dict(), f"./models/discriminator_{epoch}.pth")

    def test(self, epoch):

        G = self.generator(self.latent_dim).to(self.device)
        D = self.discriminator(self.latent_dim).to(self.device)

        G.eval()
        D.eval()

        with torch.no_grad():

            citerion = nn.BCELoss()

            
                
            g_loss = 0
            d_loss = 0
           
            for x, _ in self.test_loader:
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
                x_fake = G.decoding(z_fake)
                # Encoder
                x_true = x.float().to(self.device)
                z_true = G.encoding(x_true).view(self.batch_sie, self.latent_dim, 1, 1)
                out_true = D(x_true + noise1, z_true)
                out_fake = D(x_fake + noise2, z_fake)
                
                # losses d
                loss_d_real = citerion(out_true.view(out_true.size(0),1), y_true) 
                loss_d_fake = citerion(out_fake.view(out_fake.size(0),1), y_fake) 
                loss_d = .5 * ( loss_d_real + loss_d_fake )
                # losses g
                loss_gencoding =  F.binary_cross_entropy(out_true.view(out_true.size(0),1), y_true)
                loss_gdecoding = F.binary_cross_entropy(out_fake.view(out_fake.size(0),1), y_true) 
                loss_g = .5 * ( loss_gdecoding + loss_gencoding )
            

                d_loss += loss_d.item()
                g_loss += loss_g.item()

            if epoch % 10 == 0:
                status = f"Epoch:{epoch}/{self.num_epochs}, Total D_Loss:{loss_d.item():.5f}, total G_Loss:{loss_g.item():.5f}, D(x):{out_true.mean().item():.5f}, D(G(x)):{out_fake.mean().item():.5f}"
                print(status)

                sub_status = f"Test Epoch {epoch}, Avg Discriminator Loss: {(d_loss/len(self.train_loader)):.7f}, Avg Generator Loss: {(g_loss/len(self.train_loader)):.7f}"
                print(sub_status)

            if epoch % 30 == 0: 
                vutils.save_image(torch.cat([x_true.data[:16], x_fake.data[:16]]), f'./reconstruction/test/{epoch}.png')   


    def anomaly_detector(self, x):

        """
        This method mesure how anomalous a data point x is 
        Samples of "larger" values of A a more likely to be anomalous 
        for perfectly non anoumalous data point x : A ~ .5 * log(D(x)) which 
        ranges in (0, 1) 
        """

        G = self.generator(self.latent_dim).to(self.device)
        D = self.discriminator(self.latent_dim).to(self.device)

        G.eval()
        D.eval()

        with torch.no_grad():
            y_true = Variable(torch.ones(x.size(0), 1)).to(self.device)
            x_true = x.float().to(self.device)
            z_true = G.encoding(x_true).view(self.batch_sie, self.latent_dim, 1, 1)
            out_true = D(x_true, z_true)
            # computing the score function : mesure of data anomaly
            g_output = G(x_true)
            Lg = F.mse_loss(g_output, x_true)
            Ld = F.binary_cross_entropy(out_true.view(out_true.size(0),1), y_true)
            A = .5 * (Lg + Ld)
            status = f"The score function of this input is : {A:.5f} | reconstruction delta: {Lg:.5f} | discriminator prediction : {Ld:.5f}"
            print(status)



        

