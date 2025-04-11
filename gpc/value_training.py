from dataclasses import dataclass
from flax.struct import dataclass as fdataclass
from flax import nnx
import jax.numpy as jnp

@dataclass
class FviDataset:
    """Data class to hold the data for fitted value iteration."""
    obs: jnp.ndarray
    targets: jnp.ndarray

# @fdataclass
# class ValueApproximation:
#     model: nnx.Module
#     normalizer: nnx.BatchNorm
#
#     # TODO: Add save and load functions (?)
