from util import *
import torch
import numpy as np
import os
import h5py as h5
import sys


if __name__ == "__main__":

    ORDER = int(sys.argv[1])
    device = f"cuda:{int(sys.argv[2])}"
    idx = int(sys.argv[3])
    
    EPOCHS = 25
    NGenerators = int((ORDER*(ORDER-1))/2)
    
    
    data = torch.randn(int(1e6),ORDER)
    psi = lambda x:  torch.sum(x**2,axis = 1)
    
    
    LATENT_DIM = ORDER
    NUM_GENERATORS = int((ORDER*(ORDER-1))/2)
    BATCH_SIZE = 1024
    
    all_loss_closure = []
    all_loss_collapse = []
    all_loss_orthogonal = []
    all_generators = []

    for i in range(5): # This should be 5
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
                    eps=-1 # We want it to go to the lowest possible setting.
                   )
        generators = np.concatenate([np.array(group.algebra.detach().cpu())[:,:,None] for group in model_symmetry.group],-1)
        all_generators.append(generators[...,None])

        all_loss_closure.append(np.array(loss_S_closure)[:,None])
        all_loss_collapse.append(np.array(loss_S_collapse)[:,None])
        all_loss_orthogonal.append(np.array(loss_S_orth)[:,None])

    all_loss_closure = np.concatenate(all_loss_closure, -1)
    all_loss_collapse = np.concatenate(all_loss_collapse, -1)
    all_loss_orthogonal = np.concatenate(all_loss_orthogonal, -1)
    all_generators = np.concatenate(all_generators,-1)

    
    result_save_path = "./ConvergenceTest/"
    SAVE_PATH = os.path.join(result_save_path,f"SO_{ORDER}_convergence_test_idx_{int(sys.argv[2])}_{idx}.h5")
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
    file = h5.File(os.path.join(SAVE_PATH),"w")
    history_group = file.create_group("History")
    history_group.create_dataset(name="Closure",data=all_loss_closure,compression="lzf")
    history_group.create_dataset(name="Orthogonality",data=all_loss_orthogonal,compression="lzf")
    history_group.create_dataset(name="Collapse",data=all_loss_orthogonal,compression="lzf")


    generator_group = file.create_group("generator")
    generator_group.create_dataset(name="generator",data=all_generators,compression="lzf")
    
    file.close()