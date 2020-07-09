import streamlit as st
import torch 
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from  PIL import Image,ImageFilter





n_classes = 10
latent_size = 64


st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;

            }
            img{
                border-radius:10px;
                
            }

            </style>''', unsafe_allow_html=True)



class Generator_DCGAN(nn.Module):
    def __init__(self):
        super(Generator_DCGAN, self).__init__()

        
        self.init_size = 28 // 4

        output = 64 * self.init_size ** 2

        self.label_encoder = nn.Embedding(n_classes,output)


        self.l1 = nn.Linear(64,output)


        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), # 32 , 128 , 7, 7 
            nn.Upsample(scale_factor=2), # 32,128,14,14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),#32,128,28,28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1), # 32,1,28,28
            nn.Tanh(),
        )

    def forward(self,noise,labels):
        noise = self.l1(noise)
        noise = noise.view(noise.shape[0], 64, self.init_size, self.init_size)

        labels = self.label_encoder(labels)
        labels = labels.view(labels.shape[0], 64, self.init_size, self.init_size)

        input = torch.cat((noise,labels),1)

        
        img = self.conv_blocks(input)
        return img

# @st.cache(hash_funcs={torch._C._TensorBase: my_hash_func})
def get_model():

    G  = Generator_DCGAN()

    G.load_state_dict(torch.load('G_CDCGAN_epochs_100.pt',map_location=torch.device('cpu')))

    G.eval()

    return G


def denorm(x):
    
    out = (x + 1) / 2


    return out.clamp(0, 1)


def get_image(num):


    sample_vectors = torch.randn(1, latent_size)
    sample_labels = torch.LongTensor([num])


    model = get_model()

    with torch.no_grad():

        out = model(sample_vectors,sample_labels)

        out = denorm(out).view(28,28)

        out = out.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

        return out






st.title('MNIST Image Generator')

num =   st.text_input(label = 'enter numerical values',value= '1')

num = num.strip()


if len(num) == 1:

        image  =  get_image(int(num)).numpy()

        image = Image.fromarray(image)

        st.write(f'### Number to Generate : {num} ')
        st.image(image,width = 200)

elif len(num) > 1: 
    #123    
    num_ls = []

    for i in range(len(num)):
        
        image =  get_image(int(num[i]))

        num_ls.append(image)    

    st.write(f'No of digits {len(num_ls)}')

    image = torch.cat(num_ls,-1).numpy()

    image = Image.fromarray(image)

    st.image(image,width = 100 * len(num_ls))

else:
    st.write('please enter a number')
   











