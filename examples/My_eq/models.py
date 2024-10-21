from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Inverse(ForwardBVP):
    def __init__(self, config,x_star,y_star):
        super().__init__(config)
        self.x_star=x_star
        self.y_star=y_star
        self.phi_max = .7  
        self.l_int = .03   
        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))
        # self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, x, y):
        z = jnp.stack([x, y])
        _, u = self.state.apply_fn(params, z)
        return u
    def laplacian_2d(self,f, x, y):
        dfdx = jax.grad(f, argnums=0)
        d2fdx2 = jax.grad(dfdx, argnums=0)
        
        dfdy = jax.grad(f, argnums=1)
        d2fdy2 = jax.grad(dfdy, argnums=1)
        
        return d2fdx2(x, y) + d2fdy2(x, y)
    def compute_phi_h_phi_e(self,u):
        phi_w = u[0]
        phi_d = u[1]

        # Compute phi_h and phi_e
        phi_h = ((1 - phi_w) + phi_d) / 2
        phi_e = 1 - phi_w - phi_h

        return phi_h, phi_e, phi_d
    def compute_mu_d(self,phi_h, phi_e, phi_d, x, y):
        # Compute laplacian of phi_d
        lap_phi_d = self.laplacian_2d(lambda x, y: phi_d, x, y)
        
        # Compute mu_d
        mu_d = -phi_e + phi_h * (self.phi_max - phi_h) * (self.phi_max - 2 * phi_h) - (self.l_int**2) * lap_phi_d
        return mu_d     
    def compute_residual(self,mu_d, phi_h, phi_e, para, x, y):
        # Laplacian of mu_d
        lap_mu_d = self.laplacian_2d(lambda x, y: mu_d, x, y)

        # Compute residual
        residual = lap_mu_d + 2 * (para * phi_e - phi_h)
        return residual                                  
    def r_net(self, params, x, y):
        u = self.u_net(params, x, y)
        # Compute phi_h, phi_e
        phi_h, phi_e, phi_d = self.compute_phi_h_phi_e(u)
        
        # Compute mu_d
        mu_d = self.compute_mu_d(phi_h, phi_e, phi_d, x, y)
        
        # Compute residual
        para = u[2]  # para is the third component of u
        res = self.compute_residual(mu_d, phi_h, phi_e, para, x, y)                          
        return res

    # @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        "Compute residuals and weights for causal training"
        # Sort time coordinates
        t_sorted = batch[:, 0].sort()
        r_pred = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred**2, axis=1)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    # @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        u_pred = vmap(self.u_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        ics_loss = jnp.mean((batch[:,2:] - u_pred[:,:2]) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict

    # @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.x_star, self.y_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )
            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    # @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.x_star, self.y_star)[:,:2]
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class InverseEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.x_star, self.model.y_star)[:,:2]
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        if self.config.logging.log_nonlinearities:
            layer_keys = [
                key
                for key in state.params["params"].keys()
                if key.endswith(
                    tuple(
                        [f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]
                    )
                )
            ]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params["params"][key]["alpha"]

        return self.log_dict
