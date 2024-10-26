from itertools import chain
from typing import List, NamedTuple, Optional, Union, Tuple
from flax import linen as nn
from flax.core.frozen_dict import freeze
import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation, Updates

class CustomOptimizerState(NamedTuple):
    count: jnp.ndarray  # Counter for the number of updates
    exp_avg: Updates  # First moment (moving average of gradients)
    exp_avg_sq: Updates  # Second moment (moving average of squared gradients)

def custom_optimizer(
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0
) -> GradientTransformation:
    """
    Constructs a custom optimizer similar to Adam.
    
    Args:
        learning_rate (float): The learning rate for the optimizer.
        beta1 (float): Decay rate for the first moment (momentum) estimates.
        beta2 (float): Decay rate for the second moment estimates.
        eps (float): A small epsilon value for numerical stability.
        weight_decay (float): Optional weight decay for regularization.

    Returns:
        GradientTransformation: An optax-style optimizer.
    """
    
    def init_fn(params):
        # Initialize moving averages to zero and counter to zero
        exp_avg = jax.tree_map(jnp.zeros_like, params)  # First moment (gradient)
        exp_avg_sq = jax.tree_map(jnp.zeros_like, params)  # Second moment (squared gradient)
        count = jnp.array(0)  # Counter
        return CustomOptimizerState(count=count, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

    def update_fn(
        gradients_list: List[Updates],
        state: CustomOptimizerState,
        params: Updates,
        weights: List[float]
    ):
        """
        Update function to compute a layerwise update direction based on the gradients
        from multiple losses.

        Args:
            gradients_list (List[Updates]): List of gradients for each loss.
            state (CustomOptimizerState): Current state of the optimizer.
            params (Updates): Parameters of the model.
            weights (List[float]): Weights for each loss function.

        Returns:
            Tuple[Updates, CustomOptimizerState]: Parameter updates and new optimizer state.
        """
        count = state.count + 1

        # Ensure weights match the number of losses
        if len(weights) != len(gradients_list):
            raise ValueError("Mismatch between number of weights and number of gradient lists.")

        def solve_layer(*grads, weights):
            # Step 3a: Flatten each gradient matrix and stack into a matrix G
            G = jnp.stack([g.flatten() for g in grads], axis=0)  # Shape (num_losses, flattened_layer_dimension)
            
            # Step 3b: Solve for g_c: G_inv @ weights, using pseudo-inverse for stability
            eps = 1e-8  # Small constant for stability
            G_inv = jnp.linalg.pinv(G + eps * jnp.eye(G.shape[0]))  # Regularized inverse
            g_c_flat = G_inv @ weights  # Shape (flattened_layer_dimension, 1)
            
            # Step 3c: Reshape back to the original shape of each gradient
            g_c = g_c_flat.reshape(grads[0].shape)  # Reshape to match the shape of the individual gradients
            return g_c
        
        # Calculate combined gradient direction for each layer using the provided gradients and weights
        gc = jax.tree_multimap(lambda *g: solve_layer(*g, weights=weights), *gradients_list)

        # Update first and second moments
        exp_avg = jax.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state.exp_avg, gc)
        exp_avg_sq = jax.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g ** 2), state.exp_avg_sq, gc)

        # Compute bias-corrected estimates
        bias_correction1 = 1 - beta1 ** count
        bias_correction2 = 1 - beta2 ** count
        corrected_exp_avg = jax.tree_map(lambda m: m / bias_correction1, exp_avg)
        corrected_exp_avg_sq = jax.tree_map(lambda v: v / bias_correction2, exp_avg_sq)

        # Compute parameter updates with weight decay and learning rate scaling
        updates = jax.tree_map(
            lambda m, v, p: -learning_rate * m / (jnp.sqrt(v) + eps) - weight_decay * p,
            corrected_exp_avg, corrected_exp_avg_sq, params
        )

        # Return the updated parameters and new optimizer state
        new_state = CustomOptimizerState(count=count, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        return updates, new_state

    return GradientTransformation(init_fn, update_fn)
