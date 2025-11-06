############################################
# Author: Diptarko Choudhury
############################################

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gc
import math
import seaborn as sns

def exp_approx(x,order = 10):
    """
    Function to find the tensor exponential using the taylor series aprroximation upto
    a order of order(int).
    """
    result = 0
    term = torch.eye(n = x.shape[0],device = x.device)
    result = result + term
    for i in range(1,order+1):
        term = torch.mm(term,x)/i
        result = result + term
    return result

class GeneratorLatent(nn.Module):
    
    """
    Class to store a single Generator.
    """
    
    def __init__(self, num_features):
        super().__init__()    
        self.num_features = num_features
        self.algebra = torch.nn.Parameter(torch.empty((num_features,num_features)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.algebra, a=math.sqrt(5))
    
#--------------------------------------------------------------------------------------


class GroupLatent(nn.Module):
    def __init__(self,num_features, num_generators, LOSS_MODE = "MAE"): ## MAE works better than MSE but takes longer to converge and gives sparser generators
        super().__init__()    
        self.num_generators = num_generators
        self.LOSS_MODE = LOSS_MODE
        self.num_features = num_features
        self.group = nn.ModuleList([GeneratorLatent(self.num_features) for i in range(self.num_generators)])
        self.reset_parameters()
        self.criterion_cos = nn.CosineSimilarity(dim=0)
        
    def reset_parameters(self) -> None:
        for generator in self.group:
            generator.reset_parameters()
    
    def forward(self, theta, x, order = 10):
        transformed = x.clone() # This will keep transfroming
        for i,generator in enumerate(self.group):
            inter = 0
            term = torch.eye(x.shape[-1],device=x.device)
            inter = inter + term.expand(x.shape[0],-1,-1)
            for k in range(1,order+1,1):
                THETA = ((theta[i])**k)[:,None,None].expand(x.shape[0],x.shape[-1],x.shape[-1]) #Preparing the dimensions
                term = (term@generator.algebra)/k
                inter = inter + THETA*term.expand(x.shape[0],-1,-1)

            transformed = torch.bmm(inter,transformed[:,:,None]).squeeze()
        return transformed
        
    
    def collapse_loss(self):
        
        loss = 0
        zero = torch.zeros((self.num_features,self.num_features),device = self.group[0].algebra.device)

        for generator in self.group:
            zero_operator = exp_approx(zero)
            operator = exp_approx(generator.algebra)
            loss = loss + torch.abs(self.criterion_cos(zero_operator.ravel()[:,None],operator.ravel()[:,None])).squeeze()
        
        return loss
    
    def orthogonal_loss(self):
        
        loss = 0
        
        for i,generator1 in enumerate(self.group):
            for j,generator2 in enumerate(self.group):
                if i!=j:
                    loss = loss + torch.abs(self.criterion_cos(generator1.algebra.ravel()[:,None],generator2.algebra.ravel()[:,None])).squeeze()
        
        return loss/2
    
class MLP(nn.Module):
    def __init__(self,feature_size, feature_multipier, number_of_blocks, normalise = True):
        super(MLP,self).__init__()
        
       
        self.layers = nn.ModuleList([MLP.block(feature_size*feature_multipier) for i in range(number_of_blocks)])
        self.output = nn.Linear(feature_size*feature_multipier,feature_size,bias=False)
        if normalise:
            self.tanh = nn.Tanh()
        else:
            self.tanh = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.tanh(self.output(x))
        return x
    
    @staticmethod
    def block(features):
        return nn.Sequential(nn.LazyLinear(features),
                             # nn.BatchNorm1d(features),
                             nn.GELU())
    
class Operator(nn.Module):
    def __init__(self,
                 feature_size,
                 num_generators,
                 feature_multipier,
                 number_of_blocks,
                 normalise_enc,
                 normalise_dec,
                 LOSS_MODE = "MAE"
                ):
            super(Operator,self).__init__()


            self.encoder = MLP(feature_size=feature_size,feature_multipier=feature_multipier,number_of_blocks=number_of_blocks, normalise=normalise_enc)
            self.decoder = MLP(feature_size=feature_size,feature_multipier=feature_multipier,number_of_blocks=number_of_blocks, normalise=normalise_dec)
            self.symmetry = GroupLatent(num_features=feature_size,
                                        num_generators=num_generators,
                                        LOSS_MODE=LOSS_MODE
                                       )

    def forward(self, Z, theta, order = 10):
        enc = self.encoder(Z)
        trans = self.symmetry(theta=theta,x=enc,order = order)
        dec = self.decoder(enc)
        dec_trans = self.decoder(trans)
        return enc, trans, dec, dec_trans
    
    def orthogonal_loss(self):
        return self.symmetry.orthogonal_loss()
    
    def collapse_loss(self):
        return self.symmetry.collapse_loss()


def make_transform(data_points,
                   transformation_function, ## Function to make the transformation
                   inverse_tranformation,
                   original_oracle
                  ):

    data_point_transformed = transformation_function(data_points)

    original_mass = original_oracle(data_points)

    transformed_oracle = lambda x: original_oracle(inverse_transformation(x))

    return data_point_transformed, original_mass, transformed_oracle
    
    

    
def train_for_hidden_symmetry(train_dataloader,
                              visualisation_set,
                              optimiser_symmetry,
                              model_symmetry,
                              psi,
                              NUM_GEN,
                              device,
                              NEpochs,
                              verbose= True,
                              ):

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    loss_S_closure = []
    loss_S_orth = []
    loss_S_collapse = []
    
    loss_space = []
    loss_density = []
    
    for i in tqdm(range(NEpochs)):
        
        loss_S_closure_ = 0
        loss_S_orth_ = 0
        loss_S_collapse_ = 0
    
        loss_space_ = 0
        loss_density_ = 0
        
        for _,perturbed in train_dataloader:
            
            perturbed = perturbed.to(device)
            
            optimiser_symmetry.zero_grad()
            
            theta = [(2*torch.rand(perturbed.shape[0],device = device) - 1) for i in range(NUM_GEN)]  #Sampling
    
            enc, trans, dec, dec_trans = model_symmetry(Z = perturbed,
                                                        theta = theta)
            
            
            mass_original = psi(perturbed).detach()
            mass_transformed = psi(dec_trans)
            
            loss1 = criterion_mse(mass_original,mass_transformed)
            loss2 = model_symmetry.collapse_loss()
            loss3 = model_symmetry.orthogonal_loss()
            loss4 = criterion_mse(perturbed,dec)
            
            
            loss = (loss1 + loss2 + loss3) + loss4
            loss.backward()
            optimiser_symmetry.step()
            
            loss_S_closure_ += loss1.item()
            loss_S_collapse_ += loss2.item()
            loss_space_ += loss4.item()
            
                
        loss_S_closure_ /= len(train_dataloader)
        loss_S_collapse_ /= len(train_dataloader)
        loss_space_ /= len(train_dataloader)
        
        
        loss_S_closure.append(loss_S_closure_)
        loss_S_collapse.append(loss_S_collapse_)
        
        loss_space.append(loss_space_)

        if i%100 ==0 and verbose:
            print(f"EPOCH {i} complete")
            print("=====================")
            print("Symmetry Closure Loss ",loss_S_closure_)
            print("Symmetry Orthogonality Loss ",loss_S_orth_)
            print("Symmetry Collapse Loss ",loss_S_collapse_)
            print("Space Inversion Loss ",loss_space_)
            print("Total loss",(loss_S_closure_ + loss_S_orth_ + loss_S_collapse_ + loss_space_))

    
            pred = model_symmetry.encoder(visualisation_set.to(device)).cpu().detach().numpy()
            plt.scatter(x=pred[:, 0], y=pred[:, 1],s=0.5)
            plt.show()


    return model_symmetry