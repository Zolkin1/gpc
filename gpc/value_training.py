from dataclasses import dataclass
from flax.struct import dataclass as fdataclass
from flax import nnx
import jax.numpy as jnp
import jax
from typing import Tuple
from gpc.architectures import ValueMLP
from gpc.envs import TrainingEnv
import optax


def loss_fn(
        model: nnx.Module,
        obs: jax.Array,
        value_targets: jax.Array,
) -> jax.Array:
    """Fitted Value Iteration loss"""
    pred = model(obs)
    # print(f"pred shape: {pred.shape}")
    # print(f"val targs shape: {value_targets.shape}")
    # print(f"error size: {(pred - value_targets).shape}")
    # print(f"error: {pred - value_targets}")
    return 0.5 * jnp.mean(jnp.square(pred - value_targets))

def approximate_value(
        observations: jax.Array,
        value_targets: jax.Array,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch_size: int,
        num_epochs: int,
        rng: jax.Array,
) -> jax.Array:
    """Approximate the value function with a neural network using fitted value iteration."""

    num_data_points = observations.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    print(f"obs size: {observations.shape}")
    print(f"target size: {value_targets.shape}")

    def _train_step(
            model: nnx.Module,
            optimizer: nnx.Optimizer,
            rng: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Perform a gradient descent step on a batch of data."""
        # Get a random batch of data
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(
            batch_rng, (batch_size,), 0, num_data_points
        )

        # Get the observation and value targets
        batch_obs = observations[batch_idx] #observations[batch_idx:batch_idx + 50]
        batch_target = value_targets[batch_idx] #value_targets[batch_idx:batch_idx + 50]

        # print(f"batch obs: {batch_obs}")
        # print(f"batch target: {batch_target}")

        # print(f"batch obs size: {batch_obs.shape}")
        # print(f"batch target size: {batch_target.shape}")

        # Compute the loss and its gradient
        loss, grad = nnx.value_and_grad(loss_fn)(
            model, batch_obs, batch_target
        )

        # Update the optimizer and model parameters in-place via flax.nnx
        optimizer.update(grad)

        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)]))

        return rng, loss, grad_norm

    # for i in range(num_batches * num_epochs): take a training step
    # @nnx.scan
    # def _scan_fn(carry: Tuple, i: int) -> Tuple:
    #     model, optimizer, rng = carry
    #     rng, loss = _train_step(model, optimizer, rng)
    #     jax.debug.print("Epoch: {}, loss: {}", i, loss)
    #     return (model, optimizer, rng), loss
    #
    # _, losses = _scan_fn(
    #     (model, optimizer, rng), jnp.arange(num_epochs)
    # )

    losses = jnp.zeros(1)
    for i in range(num_epochs):
        rng, losses, grad_norm = _train_step(model, optimizer, rng)
        if i % 20 == 0:
            jax.debug.print("Epoch: {}, loss: {}. grad norm: {}", i, losses, grad_norm)
            # Check
            # pred = model(observations[0])
            # print(pred)
            # print(observations[0])
            # print(value_targets[0])

    return losses  # losses[-1]

# @nnx.jit
def jit_value_fit(
        Vnet: nnx.Module,
        optimizer: nnx.Optimizer,
        obs: jax.Array,
        targets: jax.Array,
        batch_size: int,
        num_epochs: int,
) -> jax.Array:
    # Reshape for fitting
    # y = obs #obs.reshape(-1, obs.shape[-1])
    y = obs.reshape(-1, obs.shape[-1])
    print(f"y shape {y.shape}")
    V = targets.reshape(-1, 1)
    print(f"V shape {V.shape}")

    print("Normalization complete")

    rng = jax.random.key(0)

    return approximate_value(
        observations=y,
        value_targets=V,
        optimizer=optimizer,
        model=Vnet,
        batch_size=batch_size,
        num_epochs=num_epochs,
        rng=rng,
    )

class Fvi:
    obs: jnp.ndarray
    targets: jnp.ndarray
    net: nnx.Module
    optimizer: nnx.Optimizer
    normalizer: nnx.BatchNorm
    batch_size: int
    num_epochs: int
    learning_rate: float
    V_mean: jax.Array
    V_std: jax.Array

    def __init__(self, obs_size: int, obs: jnp.ndarray, targets: jnp.ndarray,
                 batch_size: int, num_epochs: int):
        self.obs = obs
        self.targets = targets

        self.net = ValueMLP(
            observation_size=obs_size,
            hidden_layers=[32, 32],
            rngs=nnx.Rngs(0),
        )
        self.normalizer = nnx.BatchNorm(
            num_features=obs_size,
            momentum=0.1,
            use_bias=False,
            use_scale=False,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )

        self.learning_rate = 1e-3

        self.optimizer = nnx.Optimizer(self.net, optax.adamw(self.learning_rate))

        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self) -> jax.Array:
        # Normalize targets
        self.V_mean = jnp.mean(self.targets)
        self.V_std = jnp.std(self.targets)
        targets = (self.targets - self.V_mean) / self.V_std

        # Normalize Observations
        normalize_observations = True
        y = self.normalizer(self.obs, use_running_average=not normalize_observations)

        # Fit
        value_loss = jit_value_fit(self.net,
                                   self.optimizer, y, targets,
                                   self.batch_size, self.num_epochs)
        value_loss.block_until_ready()

        return value_loss

    def approximate(self, obs: jax.Array) -> jax.Array:
        # Normalize inputs
        y = self.normalizer(obs, use_running_average=True)

        # Evaluate network
        out = self.net(y)

        # Denormalize
        return out * self.V_std + self.V_mean
