import torch
#from torch.autograd import Variable
#import torch.nn as nn
#import torch.optim as optim
#from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout
#from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

## Loading the data of the time domain (after preprocessing)
class ParkinsonsDataset(Dataset):
    def __init__(self):

        self.pathFiles = "./utils/Observations"
        #self.pathFiles = 'data/stdnorm'

        
        # self.y = pd.read_csv(pathLabels)

        #C:\Users\mhele\OneDrive\Ambiente de Trabalho\DTU\2nd year\Thesis\master-thesis\utils\Observations
        #self.label = os.listdir("./utils/Observations")
       # self.label = self.y['filenames']
        #self.y=create_data_frame() #DataFrame with two columns: "filenames"(all the files in the pathFiles directory) and "binary"(all labels for each file)
        
        
       
        #dirs = os.listdir("C:/Users/mhele/OneDrive/Ambiente de Trabalho/DTU/2nd year/Thesis/master-thesis/utils/Observations") #check this
        dirs=os.listdir("../utils/Observations")
        #C:\Users\mhele\OneDrive\Ambiente de Trabalho\DTU\2nd year\Thesis\master-thesis\utils
        all_labels=[]
        for i in range(len(dirs)):
            name_size=len(dirs[i]) #19
            index_label=name_size-5
            all_labels=np.append(all_labels,int(dirs[i][index_label]))
        
        file_dictionary={"filenames":dirs}
        
        DataFrame=pd.DataFrame(file_dictionary)
        DataFrame["binary"]=all_labels.tolist()
        self.y=DataFrame
        self.label= self.y["filenames"]


    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        tempath = self.y.iloc[index]['filenames'] # tempath is a data file name "data"
        #signalpath = os.path.join(self.pathFiles,tempath)  # signalpath is the file "./folder/data.npy"
        signalpath= "../utils/Observations/%s"
        #self.x = torch.from_numpy(np.load("./Observations/observation%02d_*.npy").astype(np.double)) # load data as double //all files whose filenames start with the word observations
        self.x=torch.from_numpy(np.load(signalpath % tempath).astype(np.double))
        #print(self.y.iloc[index]['binary'])
        return self.x, self.y.iloc[index]['binary']


