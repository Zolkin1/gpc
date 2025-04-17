import mujoco
from mujoco import mjx
import jax
from typing import Sequence

from hydrax.task_base import Task
from gpc.value_training import Fvi

class TaskValue(Task):
    """Abstract task class for tasks with value function terminal costs."""

    def __init__(self,
        mj_model: mujoco.MjModel,
        planning_horizon: int,
        sim_steps_per_control_step: int,
        trace_sites: Sequence[str] = []
    ):
        super().__init__(mj_model,
        planning_horizon,
        sim_steps_per_control_step,
        trace_sites)

    def update_value_fcn(self, value_fcn: Fvi):
        self.value_fcn = value_fcn
