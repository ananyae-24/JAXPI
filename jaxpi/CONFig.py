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
from jax.tree_util import tree_map,tree_flatten
from optax import GradientTransformation, Updates
from jax.flatten_util import ravel_pytree

class CONFigState(NamedTuple):
    mt: dict # First Moment Adam 
    vt: dict # Second Moment Adam 
    m: dict # Fisrt Moment for all the losses 
    t: jnp.ndarray # Time step 

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
    def solve_layer(weights,*grads): ### Solves for the optimal Direction 
        flat_grads = jnp.stack([g.flatten() for g in grads], axis=0)  # Flatten the grads and make (#losses, #params)
        norms = jnp.linalg.norm(flat_grads, axis=1, keepdims=True) + 1e-15 # norm is 1st dimention
        ones=jnp.ones((len(grads),1)) # Vector of ones 
        G = flat_grads / norms # normalized flat Grads
        G_inv = jnp.linalg.pinv(G) # pseudo inverse 
        g_c_flat =  G_inv @ ones # g_c 
        g_c_flat/=(jnp.linalg.norm(g_c_flat.flatten())+1e-15) # normalize g_c
        g_final = (jnp.dot(flat_grads, g_c_flat)).sum()* g_c_flat # use cosine similarity to calculate the lenght
        g_final= g_final.reshape(grads[0].shape)  # reshape the grads as the original shape 
        return g_final
    def solve_layer_flatten(weights,grads): ### Solves for the optimal Direction 
        if len(weights)!=2:
            flat_grads = grads # Flatten the grads and make (#losses, #params)
            norms = jnp.linalg.norm(flat_grads, axis=1, keepdims=True) + 1e-15 # norm is 1st dimention
            ones=jnp.ones((len(weights),1)) # Vector of ones 
            G = flat_grads / norms # normalized flat Grads
            G_inv = jnp.linalg.pinv(G) # pseudo inverse 
            g_c_flat =  G_inv @ ones # g_c 
            g_c_flat/=(jnp.linalg.norm(g_c_flat.flatten())+1e-15) # normalize g_c
            g_final = (jnp.dot(flat_grads, g_c_flat)).sum()* g_c_flat # use cosine similarity to calculate the lenght
            return g_final
        norm_1=jnp.linalg.norm(grads[0])
        norm_2=jnp.linalg.norm(grads[1])
        cos_angle=(grads[0]*grads[1]).sum()/(norm_1*norm_2)
        or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
        or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
        or_1,or_2=or_1/jnp.linalg.norm(or_1),or_2/jnp.linalg.norm(or_2)
        best_direction=(1*or_1+1*or_2)
        best_direction=best_direction/jnp.linalg.norm(best_direction)
        cos_1=jnp.dot(grads[0],best_direction)
        cos_2=jnp.dot(grads[1],best_direction)
        best_direction*=cos_1+cos_2
        return best_direction
    def apply_gradients(grad_dict, weights, state): 
        t_inc=jnp.asarray(optax.safe_int32_increment(state.t)) # Increase the count
        state._replace(t=t_inc)    
        # loss_index = t% len(losses) 
        # loss_key = losses[loss_index] 
        # t_i[loss_key]+=1
        # Update biased first moment estimate
        new_m=state.m # get moments for each loss function
        new_m={ loss : tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, new_m[loss], grad_dict[loss]) for loss in losses} # Update the moments exp average 
        m_hat = { loss: tree_map(lambda m: m / (1 - (beta1 ** t_inc)), new_m[loss]) for loss in losses} # bias correction for each moment 
        weights_list = [weights[key] for key in losses]
        flattened_m_hat=[]
        tree_structure=None
        for loss in losses:
            flat_vector,tree_structure=ravel_pytree(m_hat[loss])
            flattened_m_hat.append(flat_vector)
        flattened_m_hat = jnp.array(flattened_m_hat)
        flattened_m_hat = flattened_m_hat
        final_grad=tree_structure(solve_layer_flatten(weights_list,flattened_m_hat))
        # final_grad= tree_map(
        #         lambda *grads: solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
        #         *[m_hat[key] for key in losses]  # Unpack gradients
        #     ) # calculate the m^ using the conflict free direction 
        fake_m= tree_map(lambda m: m*(1-beta1**t_inc),final_grad)
        fake_grads=tree_map(lambda m,g:( m-beta1*g)/(1-beta1),fake_m,state.mt)
        new_vt=tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2),state.vt,fake_grads)
        v_hat=tree_map(lambda v: v / (1 - beta2 ** t_inc), new_vt)
        # Update parameters
        updates = tree_map(lambda m, v:  m / (jnp.sqrt(v) + eps), final_grad, v_hat)

        return updates,state._replace(mt=fake_m, vt=new_vt,m=new_m,t=t_inc)
     
    # def init_first(grad_dict, weights, state):
    #     t_inc=jnp.asarray(optax.safe_int32_increment(state.t)) # Increase the count
    #     state._replace(t=t_inc)    
    #     new_m={ loss : grad_dict[loss] for loss in losses} 
    #     weights_list = [weights[key] for key in losses]
    #     flattened_m_hat=[]
    #     tree_structure=None
    #     for loss in losses:
    #         flat_vector,tree_structure=ravel_pytree(new_m[loss])
    #         flattened_m_hat.append(flat_vector)
    #     flattened_m_hat = jnp.array(flattened_m_hat)
    #     final_grad=tree_structure(solve_layer_flatten(weights_list,flattened_m_hat))
    #     # final_grad= tree_map(
    #     #         lambda *grads: solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
    #     #         *[new_m[key] for key in losses]  # Unpack gradients
    #     #     ) # calculate the m^ using the conflict free direction 
    #     fake_m=tree_map(lambda m: m*(1-beta1),final_grad)
    #     fake_grads=tree_map(lambda m,g:( m-beta1*g)/(1-beta1),fake_m,state.mt)
    #     new_vt=tree_map(lambda g: (1 - beta2) * (g ** 2),fake_grads)
    #     v_hat=tree_map(lambda v: v / (1 - beta2 ** t_inc), new_vt)
    #     updates = tree_map(lambda m,v:  m/ (jnp.sqrt(v) + eps), fake_m,new_vt)
    #     return updates,state._replace(mt=fake_m, vt=new_vt,m=new_m,t=t_inc)
    def update_fn(grads: dict, state: CONFigState  ,params: dict) -> Tuple[dict, CONFigState]:
        """Updates the parameters and state based on the gradients."""
        t_inc=jnp.asarray(optax.safe_int32_increment(state.t))
        weights=grads["weights"]
        del grads["weights"]
        # grads=tree_map(
        #     lambda g, w: g * w,  # Multiply each gradient by its corresponding weight
        #     grads, weights
        # )
        # params, new_state=jax.lax.cond(
        #     t_inc == 1,
        #     lambda: init_first(grads, weights, state),
        #     lambda: apply_gradients(grads, weights,state),
        # )
        params, new_state = apply_gradients( grads, weights,state)
        return params, new_state
    def init_states(params):
        return CONFigState(
                mt=jax.tree_map(jnp.zeros_like, params),
                vt=jax.tree_map(jnp.zeros_like, params),
                m={loss: jax.tree_map(jnp.zeros_like, params) for loss in losses},
                t=jnp.zeros([], jnp.int32)
            )
    return optax.chain(
            optax.GradientTransformation(
            init=lambda params: init_states(params),  
            update=update_fn),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
 