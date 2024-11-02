from itertools import chain
from typing import List, NamedTuple, Optional, Union

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict


from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from chex import Numeric
from jaxtyping import Array
from optax import GradientTransformation, Updates
from jax.tree_util import tree_map

class SOAPState(NamedTuple):
    count: jnp.ndarray  # type: ignore
    exp_avg: Updates
    exp_avg_sq: Updates
    GG: Updates
    Q: Updates
    m: Updates


def soap_config(
    learning_rate: optax.ScalarOrSchedule = 3e-3,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    losses:List=["ics","res"]
) -> optax.GradientTransformationExtraArgs:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/SOAP.

    Args:
        learning_rate (optax.ScalarOrSchedule): The learning rate to use.
        b1 (float, optional): Adam's beta1 parameter. Defaults to 0.95.
        b2 (float, optional): Adam's beta2 parameter. Defaults to 0.95.
        shampoo_beta (float, optional): If >= 0, use this beta for the preconditioner (`L` and `R` in paper, `GG` below)
            moving average instead of b2. Defaults to -1.
        eps (float, optional): Adam's epsilon for numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 0.0.
        precondition_frequency (int, optional): How often to update the preconditioner. Defaults to 10.
        max_precond_dim (int, optional): Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers. Defaults to 10000.
        precision (jax.lax.PrecisionLike, optional): Precision to use. Defaults to jax.lax.Precision.HIGHEST.

    Returns:
        optax.GradientTransformationExtraArgs: The SOAP optimizer.
    """
    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            shampoo_beta=shampoo_beta,
            eps=eps,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precision=precision,
            losses=losses
        ),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    losses:List=["ics","res"]
) -> GradientTransformation:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/SOAP.

    Args:
        b1 (float, optional): Adam's beta1 parameter. Defaults to 0.95.
        b2 (float, optional): Adam's beta2 parameter. Defaults to 0.95.
        shampoo_beta (float, optional): If >= 0, use this beta for the preconditioner (`L` and `R` in paper, `GG` below)
            moving average instead of b2. Defaults to -1.
        eps (float, optional): Adam's epsilon for numerical stability. Defaults to 1e-8.
        precondition_frequency (int, optional): How often to update the preconditioner. Defaults to 10.
        max_precond_dim (int, optional): Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers. Defaults to 10000.
        precision (jax.lax.PrecisionLike, optional): Precision to use. Defaults to jax.lax.Precision.H

    Returns:
        optax.GradientTransformationExtraArgs: The SOAP optimizer.
    """
    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2

    def init_fn(params: Updates) -> SOAPState:
        exp_avg = otu.tree_zeros_like(params)
        exp_avg_sq = otu.tree_zeros_like(params)
        GG = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim),
            params,
        )
        Q = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim),
            params,
        )
        m={loss: otu.tree_zeros_like(params) for loss in losses }
        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=GG,
            Q=Q,
            m=m
        )

    def init_step(
        updates: Updates,
        state: SOAPState,
        weights
    ) -> tuple[Updates, SOAPState]:
        new_m={ loss : tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.m[loss], updates[loss]) for loss in losses}
        weights_list = [weights[key] for key in losses]
        m_hat_g= tree_map(
                lambda *grads: solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
                *[new_m[key] for key in losses]  # Unpack gradients
            )
        updates=m_hat_g
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta),
            updates,
            state.GG,
        )

        new_Q = jtu.tree_map(
            lambda gg: get_orthogonal_matrix(gg),
            new_GG,
        )

        # Replace updates with zeros
        new_updates = otu.tree_zeros_like(updates)

        return new_updates, state._replace(GG=new_GG, Q=new_Q,m=new_m)

    def solve_layer(weights,*grads):
        flat_grads = jnp.stack([g.flatten() for g in grads], axis=0) 
        norms = jnp.linalg.norm(flat_grads, axis=1, keepdims=True) + 1e-15
        weights=jnp.ones_like(jnp.vstack(weights))
        weights=weights
        G = flat_grads / norms 
        G_inv = jnp.linalg.pinv(G)
        g_c_flat =  G_inv @ weights 
        g_c_flat/=(jnp.linalg.norm(g_c_flat.flatten())+1e-15)
        g_final = (jnp.dot(flat_grads, g_c_flat)).sum()* g_c_flat
        g_final= g_final.reshape(grads[0].shape)  
        return g_final
    
    def update_step(
        updates: Updates,
        state: SOAPState,
        weights
    ) -> tuple[Updates, SOAPState]:
        # Project gradients
        
        new_m=state.m
        new_m={ loss : tree_map(lambda m, g: b1 * m + (1 - b1) * g, new_m[loss], updates[loss]) for loss in losses}
        m_hat = { loss: tree_map(lambda m: m / (1 - (b1 ** state.count)), new_m[loss]) for loss in losses}
        weights_list = [weights[key] for key in losses]
        m_hat_g= tree_map(
                lambda *grads: solve_layer(weights_list,*grads),  # Pass weights_list as a fixed argument
                *[m_hat[key] for key in losses]  # Unpack gradients
            )
        updates=m_hat_g
        grad_projected = jtu.tree_map(
            lambda grad, q: project(grad, q, precision),
            updates,
            state.Q,
        )

        # Update moments
        exp_avg = otu.tree_update_moment(updates, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)

        exp_avg_projected = jtu.tree_map(
            lambda e, q: project(e, q, precision),
            exp_avg,
            state.Q,
        )

        # Project back
        norm_updates = jtu.tree_map(
            lambda e_avg, e_avg_sq, q: project_back(e_avg / (jnp.sqrt(e_avg_sq) + eps), q, precision),
            exp_avg_projected,
            exp_avg_sq,
            state.Q,
        )

        bc1 = 1 - b1**state.count
        bc2 = 1 - b2**state.count
        corr = jnp.sqrt(bc2) / bc1

        # Bias correction on the updates
        norm_updates = jtu.tree_map(
            lambda p: p * corr,
            norm_updates,
        )

        # Update the preconditioner
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates,
            state.GG,
        )

        # Update the orthogonal matrix / exp_avg_sq
        new_Q_and_exp_avg_sq = jax.lax.cond(
            state.count % precondition_frequency == 0,
            lambda: jtu.tree_map(
                lambda e, gg, q: get_orthogonal_matrix_QR(gg, q, e, precision),
                exp_avg_sq,
                new_GG,
                state.Q,
            ),
            lambda: jtu.tree_map(
                lambda e, q: (q, e),
                state.exp_avg_sq,
                state.Q,
            ),
        )
        ## Unpack the results
        new_Q = jtu.tree_map(
            lambda _, x: x[0],
            updates,
            new_Q_and_exp_avg_sq,
        )
        exp_avg_sq = jtu.tree_map(
            lambda _, x: x[1],
            updates,
            new_Q_and_exp_avg_sq,
        )

        new_state = SOAPState(
            count=state.count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=new_GG,
            Q=new_Q,
            m=new_m
        )

        return norm_updates, new_state

    def update_fn(updates: Updates, state: SOAPState, params: Optional[Updates] = None) -> tuple[Updates, SOAPState]:
        del params
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        state = state._replace(count=count_inc)
        weights=updates["weights"]
        del updates["weights"]
        updates, new_state = jax.lax.cond(
            count_inc == 1,
            lambda: init_step(updates, state,weights=weights),
            lambda: update_step(updates, state,weights),
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def update_preconditioner(
    grad: Array,
    GG: List[Union[Array, None]],
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> List[Union[Array, None]]:
    if grad.ndim == 1:
        return [lerp(GG[0], jnp.matmul(grad[:, None], grad[None, :], precision=precision), 1 - beta)]  # type: ignore

    new_GG = []
    for idx, gg in enumerate(GG):
        if gg is None:
            new_GG.append(None)
            continue

        outer_product = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        new_GG.append(lerp(gg, outer_product, 1 - beta))

    return new_GG


def project(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:  # noqa: SIM108
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (0,)),
                precision=precision,
            )
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = jnp.transpose(grad, permute_order)

    return grad


def project_back(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:  # noqa: SIM108
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (1,)),
                precision=precision,
            )
        else:
            grad = jnp.moveaxis(grad, 0, -1)

    return grad


def get_orthogonal_matrix(gg: Array) -> Union[Array, None]:
    if gg is None:
        return None

    _, eigh = jnp.linalg.eigh(gg + 1e-30 * jnp.eye(gg.shape[0]))
    return jnp.flip(eigh, axis=1)


def get_orthogonal_matrix_QR(
    GG: List[Union[Array, None]],
    Q: List[Union[Array, None]],
    exp_avg_sq: Array,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> tuple[List[Union[Array, None]], Array]:
    final_Q = []
    for ind, (m, o) in enumerate(zip(GG, Q)):
        if m is None or o is None:
            final_Q.append(None)
            continue

        est_eig = jnp.diag(
            jnp.matmul(
                jnp.matmul(o.T, m, precision=precision),
                o,
                precision=precision,
            )
        )
        sort_idx = jnp.argsort(est_eig, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
        o = o[:, sort_idx]
        power_iter = jnp.matmul(m, o, precision=precision)
        Q_new, _ = jnp.linalg.qr(power_iter)

        final_Q.append(Q_new)

    return final_Q, exp_avg_sq


def lerp(
    start: Array,
    end: Array,
    weight: Numeric,
):
    return start + weight * (end - start)


def init_conditioner(p: Array, max_precond_dim: int) -> List[Union[Array, None]]:
    if p.ndim == 1:
        return [jnp.zeros((p.shape[0], p.shape[0]))]

    return [jnp.zeros((s, s)) if s <= max_precond_dim else None for s in p.shape]