# import torch
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset


# class ImageDataset(Dataset):
    
#     def __init__(self,csv_name,folder,transform = None,label = False):
        
#         self.label = label
        
#         self.folder = folder
#         print(csv_name)
#         self.dataframe = pd.read_csv(self.folder+'/'+csv_name+'.csv')
#         self.tms = transform
        
#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self,index):
        
#         row = self.dataframe.iloc[index]
        
#         img_index = row['image_names']
      
#         image_file = self.folder + '/' + img_index
    
#         image = Image.open(image_file) 
        
        
#         if self.label:
#             target = row['emergency_or_not']
            
#             if target == 0:
#               encode  = torch.FloatTensor([1,0])
#             else:
#               encode = torch.FloatTensor([0,1])

#             return self.tms(image),encode
        
#         return self.tms(image)


# if __name__ == '__main__':

#     dataset = ImageDataset()



'''
since my laptop is not powerful enough I would be using the mnist dataset
'''


from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def get_dataladers():

    tms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])


    dataset =  datasets.MNIST('./datasets/mnist',
                    train = True,
                    download = True,
                    transform = tms,
                    )

    dataloader = DataLoader(dataset,batch_size=32,shuffle= True)
    
    
    return dataloader

