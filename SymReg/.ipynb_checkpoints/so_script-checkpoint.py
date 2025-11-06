from util import *
import torch
import numpy as np
import os
import h5py as h5
import sys
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--order", type=int, default=2,
                        help="order of SO-symmetry group")
    parser.add_argument("--store_idx", type=str, default="generator",
                            help="Idx of the generator whose weights will be stored")
    
    input_args = parser.parse_args()

    ORDER = input_args.order
    idx = input_args.store_idx
    
    device = "cuda:0"
    
    EPOCHS = 25
    NGenerators = int((ORDER*(ORDER-1))/2)
    
    
    data = torch.randn(int(1e6),ORDER)
    psi = lambda x:  torch.sum(x**2,axis = 1)
    
    
    LATENT_DIM = ORDER
    NUM_GENERATORS = int((ORDER*(ORDER-1))/2)
    BATCH_SIZE = 1024
    
    model_symmetry = GroupLatent(num_features=LATENT_DIM,
                                 num_generators=NUM_GENERATORS
                                )
    model_symmetry.to(device)
    optimiser_symmetry = torch.optim.Adam(model_symmetry.parameters(), lr = 1e-3)
    
    model_symmetry, loss_S_closure, loss_S_collapse, loss_S_orth = trainer(data = data,
                psi = psi,
                model_symmetry = model_symmetry,
                optimiser_symmetry = optimiser_symmetry,
                NEpochs = EPOCHS,
                BATCH_SIZE = BATCH_SIZE,
                NUM_GENERATORS = NGenerators,
                device = device,
                eps=1e-18
               )
    
    result_save_path = "./ExperimentWeights"
    generators = np.concatenate([np.array(group.algebra.detach().cpu())[:,:,None] for group in model_symmetry.group],-1)
    SAVE_PATH = os.path.join(result_save_path,f"result_SO_{ORDER}_{idx}.h5")
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)    
    file = h5.File(os.path.join(result_save_path,f"result_SO_{ORDER}_{idx}.h5"),"w")
    generator_group = file.create_group("Group")
    history_group = file.create_group("History")
    
    generator_group.create_dataset(name="Generator", data=generators, compression="lzf")
    history_group.create_dataset(name="Closure",data=loss_S_closure,compression="lzf")
    history_group.create_dataset(name="Orthogonality",data=loss_S_orth,compression="lzf")
    history_group.create_dataset(name="Collapse",data=loss_S_collapse,compression="lzf")
    
    file.close()