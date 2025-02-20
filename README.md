# Generative Predictive Control

This repository contains code for the paper ["Generative Predictive Control: Flow
Matching Policies for Dynamic and Difficult-to-Demonstrate Tasks"](https://arxiv.org/abs/2502.13406)
by Vince Kurtz and Joel Burdick. [Video summary](https://youtu.be/mjL7CF877Ow).

This includes code for training and testing flow-matching policies on each of
the robot systems shown below:

[<img src="img/cart_pole.png" width="130">](examples/cart_pole.py)
[<img src="img/double_cart_pole.png" width="130">](examples/double_cart_pole.py)
[<img src="img/pusht.png" width="130">](examples/pusht.py)
[<img src="img/walker.png" width="130">](examples/walker.py)
[<img src="img/crane.png" width="130">](examples/crane.py)
[<img src="img/humanoid.png" width="130">](examples/humanoid.py)

Generative Predictive Control (GPC) is a supervised learning framework for
training flow-matching policies on tasks that are difficult to demonstrate but
easy to simulate. GPC alternates between generating training data with
[sampling-based predictive control](https://github.com/vincekurtz/hydrax),
fitting a generative model to the data, and using the generative model to
improve the sampling distribution.

<div align="center">
<img src="img/summary.png" width="500">
</div>

## Install (Conda)

Clone and create the conda env (first time only):
```bash
git clone https://github.com/vincekurtz/gpc.git
cd gpc
conda env create -f environment.yml
```

Enter the conda env:

```bash
conda activate gpc
```

Install the package and dependencies:

```bash
pip install -e .
```

## Examples

Various examples can be found in the [`examples`](examples) directory. For
example, to train a cart-pole swingup policy using GPC, run:

```bash
python examples/cart_pole.py train
```

This will train a flow-matching policy and save it to
`/tmp/cart_pole_policy.pkl`. To run an interactive simulation with the trained
policy, run

```bash
python examples/cart_pole.py test
```

To see other command-line options, run

```bash
python examples/cart_pole.py --help
```

## Using a Different Robot Model

To try GPC on your own robot or task, you will need to:

1. Define a [Hydrax
   task](https://github.com/vincekurtz/hydrax?tab=readme-ov-file#design-your-own-task)
   that encodes the cost function and system dynamics.
2. Define a training environment that inherits from
   [`gpc.envs.base.TrainingEnv`](gpc/envs/base.py). This must implement the
   `reset`, `get_obs`, and `observation_size` methods. For example:

```python
class MyCustomEnv(TrainingEnv):
    def __init__(self):
        super().__init__(task=MyCustomHydraxTask(), episode_length=100)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the simulator to start a new episode."""
        ...
        return new_data

    def get_obs(self, data: mjx.Data) -> jax.Array:
        """Get the observation from the simulator."""
        ...
        return jax.array([obs1, obs2, ...])

    @property
    def observation_size(self) -> int:
        """Return the size of the observation vector."""
        ...
```

Then you should be able to run `gpc.training.train` to train a flow-matching
policy, and `gpc.testing.test_interactive` to run an interactive simulation with
the trained policy. See the environments in [`gpc.envs`](gpc/envs) for examples
and additional details.

## Citation

```bibtex
@article{kurtz2025generative,
  title={Generative Predictive Control: Flow Matching Policies for Dynamic and Difficult-to-Demonstrate Task},
  author={Kurtz, Vince and Burdick, Joel},
  journal={arXiv preprint arXiv:2502.13406},
  year={2025},
}
```
