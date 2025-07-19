import torch
import matplotlib.pyplot as plt
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

def frobenius_normalize(A):
    return A / A.norm(p='fro')

def spectral_normalize(A):
    u, s, v = torch.linalg.svd(A, full_matrices=False)
    return A / s[0]

def polar_normalize(A):
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh
    

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
        self.eps = 1e-5
        
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
                if i<j:
                    norm_mode = frobenius_normalize
                    mat1_norm = norm_mode(generator1.algebra)
                    mat2_norm = norm_mode(generator2.algebra)
                    loss = loss + torch.sum((mat1_norm* mat2_norm)**2)        
        return loss


def trainer(data,
            psi,
            model_symmetry,
            optimiser_symmetry,
            NEpochs,
            BATCH_SIZE,
            NUM_GENERATORS,
            device,
            eps = 1e-5
           ):

    data = data.to(device)
    criterion_mse = nn.MSELoss()
    
    loss_S_closure = []
    loss_S_orth = []
    loss_S_collapse = []
    
    loss_space = []
    loss_oracle = []
    for epoch in range(NEpochs):
        
        loss_S_closure_ = 0
        loss_S_orth_ = 0
        loss_S_collapse_ = 0
    
        loss_space_ = 0
        loss_oracle_ = 0
        
        index = torch.randperm(data.shape[0])
        for i in tqdm(range(0,(data.shape[0]//BATCH_SIZE)+1)):
            z = data[index[i*BATCH_SIZE:(i+1)*BATCH_SIZE],:]
            
            optimiser_symmetry.zero_grad()
    
            theta = [(2*torch.rand(z.shape[0],device = device) - 1) for i in range(NUM_GENERATORS)]  #Sampling
            
            z_prime = model_symmetry(theta = theta ,x = z)
    
            closure_loss = criterion_mse(psi(z),psi(z_prime))
            collapse_loss = model_symmetry.collapse_loss()
            orthogonal_loss = model_symmetry.orthogonal_loss()
            
            
            # loss_S = closure_loss + collapse_loss + orthogonal_loss
            loss_S = closure_loss + orthogonal_loss
            
            loss_S.backward()
    
            optimiser_symmetry.step()
            
        
            
            loss_S_closure_ += closure_loss.item()
            loss_S_orth_ += orthogonal_loss.item()
            loss_S_collapse_ += collapse_loss.item()
        
        loss_S_closure_ /= (data.shape[0]//BATCH_SIZE)+1
        loss_S_orth_ /= (data.shape[0]//BATCH_SIZE)+1
        loss_S_collapse_ /= (data.shape[0]//BATCH_SIZE)+1
        
        
        loss_S_closure.append(loss_S_closure_)
        loss_S_collapse.append(loss_S_collapse_)
        loss_S_orth.append(loss_S_orth_)
        
        
        print(f"EPOCH {epoch} complete")
        print("=====================")
        print("Symmetry Closure Loss ",loss_S_closure_)
        print("Symmetry Orthogonality Loss ",loss_S_orth_)
        print("Symmetry Collapse Loss ",loss_S_collapse_)

        if eps> loss_S_closure_:
            print(f"Precision of {eps} achieved, terminating run")
            break
        print("=====================")
    
    return model_symmetry, loss_S_closure, loss_S_collapse, loss_S_orth

    