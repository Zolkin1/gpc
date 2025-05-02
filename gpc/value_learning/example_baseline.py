import os

import evosax

from hydrax.algs import Evosax, PredictiveSampling, MPPI, CEM
from flax import nnx

from gpc.envs import PendulumEnv
from gpc.architectures import ValueMLP
from gpc.training import train
from gpc.value_learning.value_baseline import compute_baseline, parse_value_data, extract_data, fit_value_function, value_train

if __name__ == "__main__":
    horizon = 1.0
    knots = int(horizon * 10)
    iters = 2 #5 #1

    num_compute = 1000

    # filename = os.getcwd() + "/gpc/value_learning/data/value_data_3.pkl"
    # filename = os.getcwd() + "/gpc/value_learning/data/value_data_10k.pkl"
    # filename = os.getcwd() + "/gpc/value_learning/data/pend_cma_es_2_iter_value_data_5k.pkl"
    filename = os.getcwd() + "/gpc/value_learning/data/pend_cma_es_2_iter_value_data_1k.pkl"

    env = PendulumEnv(episode_length=200)
    ctrl = Evosax(env.task,
                  evosax.Sep_CMA_ES,
                  num_samples=256,
                  elite_ratio=0.1,
                  plan_horizon=horizon,
                  num_knots=knots,
                  iterations=iters,
                  spline_type="zero")
    # ctrl = PredictiveSampling(env.task,
    #               num_samples=256,
    #               plan_horizon=horizon,
    #               noise_level=0.1,
    #               num_knots=knots,
    #               iterations=iters,
    #               spline_type="zero")
    # compute_baseline(filename, ctrl, env, num_compute)


    net = ValueMLP(
        observation_size=env.observation_size,
        hidden_layers=[32, 32],
        rngs=nnx.Rngs(0),
    )
    value_train(env=env, net=net, filename=filename, num_epochs=500, batch_size=50, print_every=50)

    # ctrl_ps = PredictiveSampling(env.task,
    #               num_samples=256,
    #               plan_horizon=horizon,
    #               noise_level=0.1,
    #               num_knots=knots,
    #               iterations=iters,
    #               spline_type="zero")
    # compute_baseline(filename, ctrl_ps, env, num_compute)
    #
    # ctrl_mppi = MPPI(env.task,
    #               num_samples=256,
    #               plan_horizon=horizon,
    #               noise_level=0.1,
    #               temperature=0.1,
    #               num_knots=knots,
    #               iterations=iters,
    #               spline_type="zero")
    # compute_baseline(filename, ctrl_mppi, env, num_compute)
    #
    # ctrl_cem = CEM(env.task,
    #               num_samples=256,
    #               plan_horizon=horizon,
    #               num_elites=10,
    #               sigma_start=0.1,
    #               sigma_min=0.01,
    #               num_knots=knots,
    #               iterations=iters,
    #               spline_type="zero")
    # compute_baseline(filename, ctrl_cem, env, num_compute)
