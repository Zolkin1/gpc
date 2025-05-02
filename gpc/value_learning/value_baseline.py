from pyexpat import model
from typing import Tuple, Any
import time
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
import jax
import jax.numpy as jnp
from flax import nnx
import os
from hydrax.alg_base import SamplingBasedController, Trajectory
from mujoco import mjx
from tensorboardX import SummaryWriter

from gpc.envs import TrainingEnv, SimulatorState


# Run a very fine traj opt with lots of samples and iterations.
# The idea is that the trajopt will find the optimal solution so that the resulting cost can be used as the value function.
def compute_value(x: mjx.Data, ctrl: SamplingBasedController, params: Any) -> (
        Tuple)[Any, Trajectory,]:
    """Given a state, compute the value function value and the associated optimal trajectory."""
    optimize_jit = jax.jit(ctrl.optimize)
    params, rollouts = optimize_jit(x, params)

    return params, rollouts

def compute_baseline(filename: str, ctrl: SamplingBasedController, env: TrainingEnv, num_compute: int) -> jax.Array:
    """Compute the value function baseline for a given controller."""

    def _single_compute(x: SimulatorState, ctrl: SamplingBasedController, params: Any, key: jax.Array) -> None:
        """Performs the computation for a single state."""
        x = x.replace(data=env.reset(x.data, key))

        params, rollouts = compute_value(x.data, ctrl, params)

        min_cost_idx = jnp.argmin(jnp.sum(rollouts.costs, axis=1))
        costs = jnp.sum(rollouts.costs, axis=1)
        J_star = costs[min_cost_idx]
        U_star = rollouts.controls[min_cost_idx]

        return J_star, U_star, x

    print("Starting data collection...")
    start_time = time.perf_counter()

    # RNG
    key = jax.random.PRNGKey(0)

    # Initial params
    params = ctrl.init_params()

    # Initial state
    x = env.init_state(key)

    keys = jax.random.split(key, num_compute)
    J_stars, U_stars, xs = jax.vmap(_single_compute, in_axes=(None, None, None, 0))(x, ctrl, params, keys)

    print(f"J_stars shape: {J_stars.shape}, U_stars shape: {U_stars.shape}")

    # Go through a grid in the statespace
    with open(filename, 'wb') as f:
        data_chunk = {'J_star': J_stars, 'U_star': U_stars, 'state': xs}
        pickle.dump(data_chunk, f)

    end_time = time.perf_counter()
    print("End of data collection.")
    print(f"Time taken: {end_time - start_time}s")


def parse_value_data(filename: str) -> None:
    """Parse the value data from a file."""
    J_star = jnp.empty(0)
    U_star = jnp.empty(0)
    x = jnp.empty(0)
    y = jnp.empty(0)
    with open(filename, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                J_star = jnp.append(J_star, data_chunk['J_star'])
                U_star = jnp.append(U_star, data_chunk['U_star'])
                # state = jnp.append(state, data_chunk['state'])
                state = data_chunk['state']
                x = jnp.append(x, state.data.qpos[0])
                y = jnp.append(y, state.data.qvel[0])
                print("Loading data...")
            except EOFError:
                break

    plt.figure()
    plt.scatter(x, y, c=J_star)
    plt.colorbar()
    plt.show()

def extract_data(filename: str, env: TrainingEnv) -> Tuple[jax.Array, jax.Array]:
    """Extract the data from a file."""
    J_star = jnp.empty(0)
    obs = jnp.empty(0)
    with open(filename, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                J_star = jnp.append(J_star, data_chunk['J_star'])
                state = data_chunk['state']
                for i in range(J_star.shape[0]):
                    data = jax.tree.map(lambda x: x[i], state.data)
                    obs = jnp.append(obs, env.get_obs(data))
            except EOFError:
                break

    obs = obs.reshape(J_star.shape[0], 3)

    print("Data extracted.")
    return J_star, obs


def fit_value_function(model: nnx.Module,
                       optimizer: nnx.Optimizer,
                       J_star: jax.Array,
                       obs: jax.Array,
                       batch_size: int,
                       num_epochs: int,
                       rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Fit a value function to the data."""
    num_data_points = J_star.shape[0]
    num_batches = max(1, num_data_points // batch_size)

    def _loss_fn(model: nnx.Module, obs: jax.Array, value_targets: jax.Array) -> jax.Array:
        """Fitted Value Iteration loss function."""
        pred = model(obs)
        value_targets = value_targets.reshape(-1, 1)
        # print(f"pred shape: {pred.shape}")
        # print(f"value_targets shape: {value_targets.shape}")
        # print(f"square shape: {jnp.square(pred - value_targets).shape}")
        # print(f"mean shape: {jnp.mean(jnp.square(pred - value_targets)).shape}")
        return 0.5 * jnp.mean(jnp.square(pred - value_targets))

    @nnx.jit
    def _train_step(model: nnx.Module, optimizer: nnx.Optimizer, rng: jax.Array) -> Tuple:
        """Take a single gradient descent step on a batch of data."""
        rng, batch_rng = jax.random.split(rng)
        batch_idx = jax.random.randint(batch_rng, (batch_size,), 0, num_data_points)
        batch_obs = obs[batch_idx]
        batch_value_targets = J_star[batch_idx]

        # print(f"batch_obs shape: {batch_obs.shape}")
        # print(f"batch_value shape: {batch_value_targets.shape}")

        loss, grad = nnx.value_and_grad(_loss_fn)(model, batch_obs, batch_value_targets)

        optimizer.update(grad)

        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)]))

        return rng, loss, grad_norm

    # Epoch loop (through all data)
        # Batch loop (through all the batches)
    @nnx.scan
    def _scan_fn(carry: Tuple, _: int) -> Tuple:
        """Scan function for the gradient descent."""
        rng_train, optimizer_train, model_train = carry
        rng_train, loss, grad_norm = _train_step(model_train, optimizer_train, rng_train)
        return (rng_train, optimizer_train, model_train), (loss, grad_norm)

    _, (losses, grad_norms) = _scan_fn((rng, optimizer, model), jnp.arange(num_batches*num_epochs))

    return losses, grad_norms

def value_train(env: TrainingEnv,
        net: nnx.Module,
        filename: str,
        learning_rate: float = 1e-3,
        num_epochs: int = 200,
        batch_size: int = 128,
        print_every: int = 20,
      )->None:
    """Train a value function approximator."""
    start_time = time.perf_counter()
    # Print some info about the policy architecture
    params = nnx.state(net, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print(f"Policy: {type(net).__name__} with {total_params} parameters")
    print("")

    # Set up the optimizer
    optimizer = nnx.Optimizer(net, optax.adamw(learning_rate))

    # Set up the policy
    normalizer = nnx.BatchNorm(
        num_features=env.observation_size,
        momentum=0.1,
        use_bias=False,
        use_scale=False,
        use_fast_variance=False,
        rngs=nnx.Rngs(0),
    )

    J_star, obs = extract_data(filename, env)
    print(f"{J_star.shape[0]} data points.")
    print(f"Training over {num_epochs} epochs.")

    # Define log directory
    log_dir = os.path.join("logs", "value_function_fitting", str(int(time.time())))
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a SummaryWriter
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be written to: {log_dir}")

    # Break the epochs into groups and print after each group
    epoch_groups = max(1, num_epochs // print_every)
    epochs_per_group = min(num_epochs, print_every)
    for i in range(epoch_groups):
        losses, grad_norms = fit_value_function(net, optimizer, J_star, obs,
                                                batch_size, epochs_per_group,
                                                jax.random.PRNGKey(0),
                                                )
        print(f"Epoch {(i+1)*min(print_every, num_epochs)}. Loss: {losses[-1]}.")
        # print(f"Loss shape: {losses.shape}. Grad norm shape: {grad_norms.shape}. Epochs per group: {epochs_per_group}")
        for j in range(losses.shape[0]):
            writer.add_scalar("grad_norm", float(grad_norms[j]), i*losses.shape[0] + j)
            writer.add_scalar("loss", float(losses[j]), i*losses.shape[0] + j)

    print("")
    print(f"Fitting complete. Loss: {losses[-1]}")


    writer.close()

    end_time = time.perf_counter()

    print(f"Fitting took {end_time - start_time} seconds.")

    if obs.shape[1] == 3:   # Just for the pendulum
        # Plot the value network to see how it looks
        n_points = 100
        theta_net = jnp.linspace(-np.pi, np.pi, n_points)
        thetadot_net = jnp.linspace(-10, 10, n_points)

        X, Y = jnp.meshgrid(theta_net, thetadot_net)

        # Now make this into observation points
        st = jnp.sin(X)
        ct = jnp.cos(X)

        obs_points = jnp.stack([ct.reshape(-1), st.reshape(-1), Y.reshape(-1)], axis=-1)
        # print(f"obs size: {obs_points.shape}")
        # print(f"st size: {st.shape}")
        # print(f"ct size: {ct.shape}")
        # print(f"Y size: {Y.shape}")

        Z_flat = net(obs_points)
        # print(f"Z_flat size: {Z_flat.shape}")
        Z = Z_flat.reshape((n_points,n_points))
        # print(f"Z size: {Z.shape}")

        plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis', extent=(X.min(), X.max(), Y.min(), Y.max()))
        max_points = min(obs.shape[0], 1000)
        theta = jnp.arctan2(obs[:max_points, 1], obs[:max_points, 0]).reshape(-1)
        theta_dot = obs[:max_points, 2]
        plt.scatter(theta, theta_dot, c=J_star[:max_points], cmap='viridis')
        plt.colorbar(label='Z Value')
        plt.xlabel('theta')
        plt.ylabel('theta dot')
        plt.title('Fit Value Function')
        plt.show()

        plt.figure()
        plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis', extent=(X.min(), X.max(), Y.min(), Y.max()))
        plt.colorbar(label='Z Value')
        plt.xlabel('theta')
        plt.ylabel('theta dot')
        plt.title('Fit Value Function')
        plt.show()

