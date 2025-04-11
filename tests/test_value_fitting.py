import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax

from gpc.value_training import Fvi, loss_fn

def test_value_regression() -> None:
    """Test fitting the value function to data."""

    # Generate data
    training_size = 100
    obs1 = jnp.linspace(-5, 5, training_size)
    obs2 = jnp.zeros(training_size)
    obs = jnp.stack([obs1, obs2], axis=1)
    targets = 10*jnp.sin(obs1)

    print(f"original obs shape: {obs.shape}")
    print(f"original targets shape: {targets.shape}")

    obs_size = 2

    # Make the approximator
    value_approximator = Fvi(obs_size=obs_size, obs=obs, targets=targets)

    # Fit the data
    loss = value_approximator.fit()
    loss.block_until_ready()

    # Plot the data
    plt.figure()
    plt.plot(obs[:,0], targets)
    plt.xlabel("Obs")
    plt.ylabel("Targets/Fit Values")

    # Plot the fit network
    test_obs = jnp.linspace(-5, 5, 200)
    test_obs2 = jnp.zeros(200)
    plt.plot(test_obs, value_approximator.approximate(jnp.stack([test_obs, test_obs2], axis=1)))
    plt.show()

    print(f"loss: {loss_fn(value_approximator.net, obs, targets)}")
    print(f"value approximation: {value_approximator.approximate(obs)}")
