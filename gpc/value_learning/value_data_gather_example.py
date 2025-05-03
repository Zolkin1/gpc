import os

import evosax

import jax
from hydrax.algs import Evosax, PredictiveSampling, MPPI, CEM
from flax import nnx

from gpc.envs import PendulumEnv
from gpc.architectures import ValueMLP
from gpc.training import train
from gpc.value_learning.value_baseline import compute_baseline, parse_value_data, extract_data
from gpc.value_learning.value_training import value_train

from value_data_gathering import gather_data

horizon = 1.0
knots = int(horizon * 10)
iters = 2 #5 #1

env = PendulumEnv(episode_length=200)
ctrl = Evosax(env.task,
              evosax.Sep_CMA_ES,
              num_samples=256,
              elite_ratio=0.1,
              plan_horizon=horizon,
              num_knots=knots,
              iterations=iters,
              spline_type="zero")

J_start, obs = gather_data(env, ctrl, 200, jax.random.PRNGKey(0))

net = ValueMLP(
    observation_size=env.observation_size,
    hidden_layers=[32, 32],
    rngs=nnx.Rngs(0),
)

value_train(env=env, net=net, J_star=J_start, obs=obs, num_epochs=500, batch_size=50, print_every=50)
