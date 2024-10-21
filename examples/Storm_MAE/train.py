import os
import time

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import scipy.io

import ml_collections
import wandb


from jaxpi.archs import PeriodEmbs, Embedding
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint
# import numpy as np 
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count
from jaxpi.samplers import BaseSampler

import models
from utils import get_dataset



class UniformSamplerFile(BaseSampler):
    def __init__(self, x, y, u_ref, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.x = x  # 1D array of x coordinates
        self.y = y  # 1D array of y coordinates
        self.u_ref = u_ref  # 2D array with two columns (val1, val2)
        assert u_ref.shape[1] == 2, "u_ref should have 2 columns."

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        """Generates data containing batch_size samples from the numpy arrays"""
        # Randomly sample indices from x, y, and u_ref
        num_samples = len(self.x)
        indices = random.randint(key, shape=(self.batch_size,), minval=0, maxval=num_samples)

        # Select x, y, u_ref based on sampled indices
        batch_x = self.x[indices]
        batch_y = self.y[indices]
        batch_u_ref = self.u_ref[indices]

        # Stack x, y, u_ref into a single array of shape (batch_size, 4)
        batch = jnp.column_stack((batch_x, batch_y, batch_u_ref))  # Each row is [x, y, u_ref_val1, u_ref_val2]

        return batch

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()
    # Get dataset
    u_ref, x, y = get_dataset()
    
    # Define residual sampler
    res_sampler = iter(UniformSamplerFile(x, y, u_ref, config.training.batch_size_per_device))
    if config.use_pi_init:
        logger.info("Use physics-informed initialization...")

        model = models.Inverse(config,x,y)
        
        state = jax.device_get(tree_map(lambda x: x[0], model.state))
        params = state.params
        u,x,y=get_dataset()
        inputs = jnp.stack([x, y]).T
        # Initialization data source
        if config.pi_init_type == "linear_pde":
            # load data
            # data = scipy.io.loadmat("data/allen_cahn_linear.mat")
            # downsample the grid and data
            u,x,y=get_dataset()
            # u = data["usol"][::10]
            # t = data["t"].flatten()[::10]
            # x = data["x"].flatten()

            # xx,yy = jnp.meshgrid(x, y, indexing="ij")
            inputs = jnp.stack([x[:,0], y[:,0]]).T
        elif config.pi_init_type == "initial_condition":
            xx,yy = jnp.meshgrid(x, y, indexing="ij")
            inputs = jnp.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
            u = jnp.tile(u.flatten(), (x.shape[0], 1))
        feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)
        coeffs, residuals, rank, s = jnp.linalg.lstsq(
            feat_matrix, jnp.concatenate([u, jnp.ones((u.shape[0], 1))], axis=1), rcond=None
        )
        print("least square residuals: ", residuals)
        config.arch.pi_init = coeffs#.reshape(-1, 1)  # Be careful, this overwrites the config file!

        del model, state, params

    # Initialize model
    model = models.Inverse(config,x,y)

    # Initialize evaluator
    evaluator = models.InverseEvaluator(config, model)

    logger.info("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.abspath(os.path.join(workdir, config.wandb.name, "ckpt"))
                # path = os.path.join(workdir, config.wandb.name, "ckpt")
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
