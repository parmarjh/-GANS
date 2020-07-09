import dataset 

from torch.autograd import Variable

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch


import numpy as np


dataloader = dataset.get_dataladers()



for image,_ in dataloader:

    print(image.size(0))


    input_size = image[0].view(-1).size()[0]

    print(input_size)

    break


class Generator(nn.Module):


    def __init__(self):

        super(Generator,self).__init__()

        def block(i,out):
            
            layers =  [nn.Linear(i,out),
                        nn.BatchNorm1d(out),
                        nn.ReLU(0.2)
                        ]

            return layers            


        self.model = nn.Sequential(
                                *block(64,128),
                                *block(128,256),
                                *block(256,784),
                                nn.Tanh()
                                )

      
    def forward(self,x):

        out = self.model(x)

        out = out.view(-1,1,28,28)    

        return out



class Descriminator(nn.Module):


    def __init__(self):

        super(Descriminator,self).__init__()

        self.model = nn.Sequential(
                                    nn.Linear(784,128),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(128,64),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(64,1),
                                    nn.Sigmoid()
                                    
                                    )
    def forward(self,x):
        x = x.view(-1,784)
        out = self.model(x)

        return out    







epochs = 100



z = torch.rand(32,64)    

g = Generator()
d = Descriminator()


loss = nn.BCELoss()
optimizer_G = torch.optim.Adam(g.parameters(), 0.00002,betas=(0.5,0.9))
optimizer_D = torch.optim.Adam(d.parameters(), 0.00002,betas=(0.5,0.9))


for i in range(epochs):
        gen_loss = []
        des_loss = []

        for image,_ in dataloader:

           

            # train the descriminator 

            real_labels = torch.FloatTensor(image.size(0),1).fill_(1.0)
            fake_labels = torch.FloatTensor(image.size(0),1).fill_(0.0)
        

            #calculate generator loss

            optimizer_G.zero_grad()
            
            z = torch.FloatTensor(np.random.normal(0, 1, (image.size(0), 64)))
            fake_images = g(z)

            g_loss = loss(d(fake_images),real_labels)

          
            g_loss.backward()

            optimizer_G.step()





            optimizer_D.zero_grad()

            # calculate loss for real images pdata
            real_out =  d.forward(image)   

            real_loss  = loss(real_out,real_labels)

            #calculating loss for fake images p
           

            fake_loss = loss(d(fake_images.detach()),fake_labels)

            d_loss =  (real_loss + fake_loss)/2

            

            d_loss.backward()

            optimizer_D.step()
   

            gen_loss.append(g_loss.item())
            des_loss.append(d_loss.item())

            



        print(f'gen loss: {np.mean(gen_loss)}  dis loss: {np.mean(des_loss)}')
        save_image(fake_images.data[:25], f"images{i}.jpg", nrow=5)