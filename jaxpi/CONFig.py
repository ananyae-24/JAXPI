from flax import struct
from itertools import chain
from typing import List, NamedTuple, Optional, Union, Tuple
from flax import linen as nn
from flax.core.frozen_dict import freeze
import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation, Updates
from functools import partial
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from optax import GradientTransformation, Updates


class CONFigState(NamedTuple):
    mt: dict
    vt: dict
    m: dict
    t: jnp.ndarray

# class CONFig:
#     def __init__(self,losses=["ics","res"], learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8,weight_decay=0.0):
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.m = {loss: None for loss in losses}  # First moment for each loss
#         self.t=0 # Time step
#         self.t_i =  {loss: 0 for loss in losses}   
#         self.num_loss=2
#         self.losses=losses
#         self.mt=None 
#         self.vt=None
#         self.weight_decay=weight_decay
        
#     def init_states(self, params):
#         return CONFigState(
#             mt=jax.tree_map(jnp.zeros_like, params),
#             vt=jax.tree_map(jnp.zeros_like, params),
#             m={loss: jax.tree_map(jnp.zeros_like, params) for loss in self.losses}
#         )
        
#     # def init_states(self, params):
#     #     self.mt=jax.tree_map(jnp.zeros_like, params)
#     #     self.vt=jax.tree_map(jnp.zeros_like, params)
#     #     self.m = {loss: jax.tree_map(jnp.zeros_like, params) for loss in self.losses}
            
#     # def init(self, params):
#     #     """Initialize optimizer state based on model parameters."""
#     #     return self.init_states(params)
#     @partial(jax.jit, static_argnums=(0,))
#     def solve_layer(self,weights,*grads):
#         flat_grads = jnp.stack([g.flatten() for g in grads], axis=0) 
#         norms = jnp.linalg.norm(flat_grads, axis=1, keepdims=True) + 1e-15
#         weights=jnp.vstack(weights)
#         weights=weights
#         G = flat_grads / norms 
#         G_inv = jnp.linalg.pinv(G)
#         g_c_flat =  G_inv @ weights 
#         g_c_flat/=(jnp.linalg.norm(g_c_flat.flatten())+1e-15)
#         g_final = jnp.dot(flat_grads, g_c_flat).sum()* g_c_flat
#         g_final= g_final.reshape(grads[0].shape)  
#         return g_final
    
#     def apply_gradients(self, params, grad_dict, weights):
#         # Update time step
#         self.t += 1
#         loss_key = self.losses[self.t % len(self.losses)]
#         self.t_i[loss_key]+=1
#         # Update biased first moment estimate
#         self.m[loss_key]=tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m[loss_key], grad_dict[loss_key])
#         m_hat = {loss: tree_map(lambda m: m / (1 - self.beta1 ** self.t_i.get(loss, 1)), self.m[loss]) for loss in self.losses}
#         weights_list = [weights[key] for key in self.losses]
#         m_hat_g= tree_map(
#             lambda *grads: self.solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
#             *[m_hat[key] for key in self.losses]  # Unpack gradients
#         )
#         g_c=tree_map(lambda m,g: ((1-self.beta1**self.t)*m-self.beta1*g)/(1-self.beta1),m_hat_g,self.mt)
#         self.mt=tree_map(lambda m,g: self.beta1*m+(1-self.beta1)*g,self.mt,g_c)
#         self.vt=tree_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * (g ** 2),self.vt,g_c)
#         v_hat=tree_map(lambda v: v / (1 - self.beta2 ** self.t), self.vt)
#         # Update parameters
#         new_params = tree_map(lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.eps), params, m_hat_g, v_hat)

#         return new_params,CONFigState(mt=self.mt, vt=self.vt, m=self.m)
#     def __call__(self):
#         def update_fn(grads: dict, state: CONFigState  ,params: dict) -> Tuple[dict, CONFigState]:
#             """Updates the parameters and state based on the gradients."""
#             weights=grads["weights"]
#             del grads["weights"]
#             self.mt,self.vt,self.m=state.mt,state.vt,state.m
#             params, new_state = self.apply_gradients(params, grads, weights)
#             return params, new_state      
#         return optax.chain(
#                     optax.GradientTransformation(
#                     init=lambda params: self.init_states(params),  
#                     update=update_fn),
#                     optax.add_decayed_weights(self.weight_decay),
#                     optax.scale_by_learning_rate(self.learning_rate),
#                 )

def CONFig(losses:List =["ics","res"],learning_rate:optax.ScalarOrSchedule = 3e-3, beta1=0.9, beta2:float=0.999, eps:float=1e-8,weight_decay: float = 0.0)-> GradientTransformation:
    def solve_layer(weights,*grads):
        flat_grads = jnp.stack([g.flatten() for g in grads], axis=0) 
        norms = jnp.linalg.norm(flat_grads, axis=1, keepdims=True) + 1e-15
        weights=jnp.ones_like(jnp.vstack(weights))
        weights=weights
        G = flat_grads / norms 
        G_inv = jnp.linalg.pinv(G)
        g_c_flat =  G_inv @ weights 
        g_c_flat/=(jnp.linalg.norm(g_c_flat.flatten())+1e-15)
        g_final = (jnp.dot(flat_grads, g_c_flat)*weights).sum()* g_c_flat
        g_final= g_final.reshape(grads[0].shape)  
        return g_final
    def apply_gradients(params, grad_dict, weights,state): 
        t_inc=jnp.asarray(optax.safe_int32_increment(state.t))
        state._replace(t=t_inc)    
        # loss_index = t% len(losses) 
        # loss_key = losses[loss_index] 
        # t_i[loss_key]+=1
        # Update biased first moment estimate
        new_m=state.m
        new_m={ loss : tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, new_m[loss], grad_dict[loss]) for loss in losses}
        m_hat = { loss: tree_map(lambda m: m / (1 - (beta1 ** state.t)), new_m[loss]) for loss in losses}
        weights_list = [weights[key] for key in losses]
        m_hat_g= tree_map(
                lambda *grads: solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
                *[m_hat[key] for key in losses]  # Unpack gradients
            )
        new_mt=state.mt
        g_c=tree_map(lambda m,g: ((1-beta1**state.t)*m-beta1*g)/(1-beta1),m_hat_g,new_mt)
        new_mt=tree_map(lambda m,g: beta1*m+(1-beta1)*g,new_mt,g_c)
        new_vt=tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2),state.vt,g_c)
        v_hat=tree_map(lambda v: v / (1 - beta2 ** state.t), new_vt)
        # Update parameters
        new_params = tree_map(lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps), params, m_hat_g, v_hat)

        return new_params,state._replace(mt=new_mt, vt=new_vt,m=new_m,t=t_inc)
    
    def update_fn(grads: dict, state: CONFigState  ,params: dict) -> Tuple[dict, CONFigState]:
        """Updates the parameters and state based on the gradients."""
        weights=grads["weights"]
        del grads["weights"]
        # grads=tree_map(
        #     lambda g, w: g * w,  # Multiply each gradient by its corresponding weight
        #     grads, weights
        # )
        params, new_state = apply_gradients(params, grads, weights,state)
        return params, new_state
    def init_states(params):
        return CONFigState(
                mt=jax.tree_map(jnp.zeros_like, params),
                vt=jax.tree_map(jnp.zeros_like, params),
                m={loss: jax.tree_map(jnp.zeros_like, params) for loss in losses},
                t=jnp.ones([], jnp.int32)
            )
    return optax.chain(
            optax.GradientTransformation(
            init=lambda params: init_states(params),  
            update=update_fn),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
 