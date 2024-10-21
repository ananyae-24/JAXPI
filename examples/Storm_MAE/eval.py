import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, x, y = get_dataset()
    # u_ref, t_star, x_star = get_dataset()

    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    # Restore model
    model = models.Inverse(config,x,y)
    ckpt_path = os.path.abspath(os.path.join(workdir, config.wandb.name, "ckpt"))
    # ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.x_star, model.y_star)
    u_pred_res=model.r_pred_fn(params, model.x_star, model.y_star)
    # XX, YY = jnp.meshgrid(x, y, indexing="ij")

    # plot
    fig, axs = plt.subplots(4, 3, figsize=(24, 16))
    # fig = plt.figure(figsize=(18, 5))
    # plt.subplot(1, 3, 1)
    p_lot=axs[0,0].scatter(x,y, c=u_ref[:,0], cmap="jet")
    fig.colorbar(p_lot,ax=axs[0,0])
    axs[0,0].set_xlabel("x")
    axs[0,0].set_ylabel("y")
    axs[0,0].set_title("Reference phi_w")
    
    p_lot=axs[0,1].scatter(x, y, c=u_pred[:,0], cmap="jet")
    fig.colorbar(p_lot,ax=axs[0,1])
    axs[0,1].set_xlabel("x")
    axs[0,1].set_ylabel("y")
    axs[0,1].set_title("Pred phi_w")
    
    p_lot=axs[0,2].scatter(x, y, c=jnp.abs(u_ref[:,0] - u_pred[:,0]), cmap="jet")
    fig.colorbar(p_lot,ax=axs[0,2])
    axs[0,2].set_xlabel("x")
    axs[0,2].set_ylabel("y")
    axs[0,2].set_title("Absolute error phi_w")
    # plt.tight_layout()

    p_lot=axs[1,0].scatter(x, y, c=u_ref[:,1], cmap="jet")
    fig.colorbar(p_lot,ax=axs[1,0])
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("y")
    axs[1,0].set_title("Reference phi_d")
    
    p_lot=axs[1,1].scatter(x, y, c=u_pred[:,1], cmap="jet")
    fig.colorbar(p_lot,ax=axs[1,1])
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("y")
    axs[1,1].set_title("Pred phi_d")
    
    p_lot=axs[1,2].scatter(x, y, c=jnp.abs(u_ref[:,1] - u_pred[:,1]), cmap="jet")
    fig.colorbar(p_lot,ax=axs[1,2])
    axs[1,2].set_xlabel("x")
    axs[1,2].set_ylabel("y")
    axs[1,2].set_title("Absolute error phi_d")

    p_lot=axs[2,0].scatter(x, y, c=u_pred[:,2], cmap="jet")
    fig.colorbar(p_lot,ax=axs[2,0])
    axs[2,0].set_xlabel("x")
    axs[2,0].set_ylabel("y")
    mean,std=str(jnp.round(u_pred[:,2].mean(),3)),str(jnp.round(u_pred[:,2].std(),3))
    print(mean,std)
    axs[2,0].set_title(f"Parameter mean: {mean[:5]} std: {std[:5]}")
    
    p_lot=axs[2,1].scatter(x, y, c=u_pred_res, cmap="jet")
    fig.colorbar(p_lot,ax=axs[2,1])
    axs[2,1].set_xlabel("x")
    axs[2,1].set_ylabel("y")
    axs[2,1].set_title(f"Residuals")
    
    p_lot=axs[2, 2].hist(jnp.abs(u_ref[:,0] - u_pred[:,0]).flatten(),500)
    # fig.colorbar(p_lot,ax=axs[2,2])
    axs[2,1].set_xlabel("Error")
    axs[2,1].set_ylabel("Count")
    axs[2,1].set_title(f"Error Phi_w Histogram")
    
    p_lot=axs[3, 0].hist(jnp.abs(u_ref[:,1] - u_pred[:,1]).flatten(),500)
    # fig.colorbar(p_lot,ax=axs[3,0])
    axs[3,0].set_xlabel("Error")
    axs[3,0].set_ylabel("Count")
    axs[3,0].set_title(f"Error Phi_d Histogram")
    axs[3,1].axis("off")
    axs[3,2].axis("off")
    print((jnp.abs(u_ref[:,1] - u_pred[:,1])>.04).sum(), jnp.abs(u_ref[:,1] - u_pred[:,1]).size)
    fig.suptitle(f"{config.wandb.name}", fontsize=16)
    plt.tight_layout()
    # Save the figure
    save_dir = os.path.abspath(os.path.join(workdir, "figures", config.wandb.name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "ac.png")
    fig.savefig(fig_path)#, bbox_inches="tight", dpi=300)
