from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import SamplingBasedController, Trajectory, SamplingParams

@dataclass
class PACParams(SamplingParams):
    """Parameters for the policy-augmented controller.

    Attributes:
        base_params: The parameters for the base controller.
        policy_samples: Control sequences sampled from the policy.
        rng: Random number generator key for domain randomization.
    """

    base_params: Any
    policy_knots: Any
    tk: jax.Array
    mean: jax.Array
    rng: jax.Array

class PolicyAugmentedController(SamplingBasedController):
    """An SPC generalization where samples are augmented by a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        super().__init__(
            base_ctrl.task,
            base_ctrl.num_randomizations,
            base_ctrl.risk_strategy,
            num_knots=base_ctrl.num_knots,
            plan_horizon=base_ctrl.plan_horizon,
            seed=0,
        )

    def init_params(self) -> PACParams:
        """Initialize the controller parameters."""
        base_params = self.base_ctrl.init_params()
        base_rng, our_rng = jax.random.split(base_params.rng)
        base_params = base_params.replace(rng=base_rng)
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.num_knots,
                self.task.model.nu,
            )
        )
        return PACParams(
            base_params=base_params,
            policy_knots=policy_samples,
            tk=base_params.tk,
            mean=base_params.mean,
            rng=base_params.rng,
        )

    def sample_knots(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample control sequences from the base controller and the policy."""
        # Samples from the base controller
        base_samples, base_params = self.base_ctrl.sample_knots(
            params.base_params
        )

        # Include samples from the policy. Assumes that these have already been
        # generated and stored in params.policy_knots.
        samples = jnp.append(base_samples, params.policy_knots, axis=0)

        return samples, params.replace(base_params=base_params)

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update the policy parameters according to the base controller."""
        base_params = self.base_ctrl.update_params(params.base_params, rollouts)
        return params.replace(base_params=base_params)

    def get_action(self, params: PACParams, t: float) -> jax.Array:
        """Get the action from the base controller at a given time."""
        return self.base_ctrl.get_action(params.base_params, t)

    def get_action_sequence(self, params: PACParams) -> jax.Array:
        """Get the action sequence from the controller."""
        timesteps = jnp.linspace(0.0, self.plan_horizon, int(self.plan_horizon/self.dt))
        return jax.vmap(self.get_action, in_axes=(None, 0))(params, timesteps)  # TODO: Make sure this returns a model.nu dimension

    def get_action_from_policy(self, knots: jax.Array, tk: jax.Array, t: jax.Array) -> jax.Array:
        """Get the action from the policy at a given time."""
        print(f"interp shape result {self.interp_func(t, tk, knots).shape}")
        u = self.interp_func(t, tk, knots)[0]  # (nu,)
        return u

    def get_action_sequence_from_policy(self, knots: jax.Array, tk: jax.Array) -> jax.Array:
        print(f"knots shape {knots.shape}")
        timesteps = jnp.linspace(0.0, self.plan_horizon, int(self.plan_horizon/self.dt))
        return jax.vmap(self.get_action_from_policy, in_axes=(None, None, 0))(knots, tk, timesteps)
