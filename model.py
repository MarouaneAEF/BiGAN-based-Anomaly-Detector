import torch 
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, z_dim, wasserstein=False):

        super().__init__()
        self.was = wasserstein

        #  x graph stack 
        self.xStack = nn.Sequential(

            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)

        )

        # z graph stack 
        self.zStack = nn.Sequential(
            nn.Conv2d(z_dim, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)

        )

        # joint (x,z) graph 

        self.xzStack = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(1024, 1, 1, stride=1, bias=False)

        )

    def x_graph(self, x):

        x = self.xStack(x)
        return x
    
    def z_graph(self, z):

        z = self.zStack(z)
        return z
    
    def xz_graph(self, xz):
        
        xz = self.xz_graph(xz) 
        return xz

    def forward(self, x, z):

        x = self.x_graph(x)
        z = self.zStack(z)
        xz = torch.cat((x,z), dim=1)
        output = self.xzStack(xz)

        if self.was:
            return output
        else:
            output =  torch.sigmoid(output) 
            return output

class Generator(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)

        self.genStack = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

             nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
             nn.BatchNorm2d(64),
             nn.LeakyReLU(0.1, inplace=True),

             nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.1, inplace=True),

             nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
             nn.BatchNorm2d(32),
             nn.LeakyReLU(0.1, inplace=True),

             nn.Conv2d(32, 3, 1, stride=1, bias=True),
            #  nn.Sigmoid()
             
        )

    def forward(self, z):

        z = self.genStack(z)
        return torch.sigmoid(z + self.output_bias)

class Encoder(nn.Module):
    def __init__(self, z_dim=32):

        super().__init__()
        self.z_dim = z_dim

        self.encStack = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, z_dim*2, 1, stride=1, bias=True),
            nn.Sigmoid()

        )

    def reparametrize(self, z):
        z = z.view(z.size(0), -1)
        mu, log_var = z[:, :self.z_dim], z[:,self.z_dim:]
        std = torch.exp(0.5* log_var)
        epsilon = torch.randn_like(std)
        

        sample = mu + epsilon*std
        return sample 

    def forward(self, x):

        x = self.encStack(x)
        z_sample = self.reparametrize(x)

        return z_sample.view(x.size(0), self.z_dim, 1, 1)



