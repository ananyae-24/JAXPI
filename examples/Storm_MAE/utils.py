import scipy.io
import pandas as pd 
import jax
import jax.numpy as jnp
from jax import vmap
import re 
import numpy as np 
def get_dataset(T=2.0, L=1, c=2, n_t=200, n_x=128):
    df1,df2=pd.read_csv("./data/Phiw.csv"),pd.read_csv("./data/Phid.csv")
    w_d=jnp.array([df1["Phiw"],df2["Phid"]]).T
    grid=jnp.array([df1["X"],df2["Y"]]).T
    return w_d, grid[:,0],grid[:,1]
