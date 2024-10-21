import scipy.io

import jax
import jax.numpy as jnp
from jax import vmap
import re 
import numpy as np 
def process_file(file):
    grid,val=[],[]
    with open(file,"r") as f:
        f.readline(),f.readline(),f.readline(),f.readline(),f.readline(),f.readline(),f.readline(),f.readline()
        while True :
            x=f.readline()
            if not x: break 
            curr=re.sub("\s+",",",str(x))
            curr=curr.split(",")
            grid.append([float(curr[0]),float(curr[1])])
            val.append(float(curr[2]))
    return np.array(grid),np.array(val)
def get_dataset(T=2.0, L=1, c=2, n_t=200, n_x=128):
    grid,val1=process_file("./data/Phiw_1_0.txt") 
    grid1,val2=process_file("./data/Phid_1_0.txt")
    grid2,gme=process_file("./data/G_me_1_0.txt")
    w_d=jnp.array([val1,val2]).T
    grid=jnp.array(grid1)
    return w_d, grid[:,0],grid[:,1]
