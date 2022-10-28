import sys
import warnings

# rcognita base
from rcognita_framework.pipelines.pipeline_blueprints import PipelineWithDefaults
from rcognita_framework.rcognita.simulator import Simulator
from rcognita_framework.rcognita.optimizers import TorchOptimizer
from rcognita_framework.rcognita.loggers import Logger
from rcognita_framework.rcognita.predictors import EulerPredictor

# config
from configs import ConfigMarsLander

from .models import ModelActorMarsLander, ModelCriticMarsLander, ModelRunningObjectiveMarsLander
from .actor import ActorMarsLander
from .animator import AnimatorMarsLander
from .critic import CriticMarsLander
from .scenario import EpisodicScenarioMarsLander
from .system import SysMarsLander

import matplotlib.animation as animation
from rcognita.utilities import on_key_press
import matplotlib.pyplot as plt
import numpy as np
import torch


class PipelineMarsLander(PipelineWithDefaults):
    config = ConfigMarsLander
    def initialize_logger(self):

        self.datafiles = [None]

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.logger = Logger(
            ["x [m]", "y [m]", "angle [rad]", "v_x [m/s]", "v_y [m/s]"], ["a [m/s^2]", "beta [rad/s]"]
        )
        self.logger.N_iterations = self.N_iterations
        self.logger.N_episodes = self.N_episodes

    def initialize_predictor(self):
        self.predictor = EulerPredictor(
            self.pred_step_size,
            self.system._compute_state_dynamics,
            self.system.out,
            self.dim_output,
            self.prediction_horizon,
        )

    def initialize_system(self):
        self.system = SysMarsLander(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[self.g],
            is_dynamic_controller=self.is_dynamic_controller,
            is_disturb=self.is_disturb,
            pars_disturb=[],
        )

    def initialize_models(self):
        self.critic_model = ModelCriticMarsLander(
                                self.dim_output,
                                self.dim_input
                            )

        self.actor_model  = ModelActorMarsLander(
                                self.dim_output,
                                self.dim_input
                            )

        self.model_running_objective = ModelRunningObjectiveMarsLander(
                                fuel_consumption_coeff=self.fuel_consumption_coeff,
                                angle_constraint_coeff=self.angle_constraint_coeff
                            )


    def initialize_optimizers(self):
        self.opt_options_actor = {
            "lr": self.learning_rate_actor,
            "weight_decay": self.weight_decay_actor
        }
        opt_options_critic = {
            "lr": self.learning_rate_critic,
            "weight_decay": self.weight_decay_critic
        }

        self.critic_optimizer = TorchOptimizer(
            opt_options_critic, iterations=1
        )

    def initialize_actor_critic(self):
        """
            TODO: fix
        """

        self.critic = CriticMarsLander(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size,
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            sampling_time=self.sampling_time,
        )

        self.actor = ActorMarsLander(
            state_to_observation=self.system.out,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            control_mode=self.control_mode,
            action_bounds=self.action_bounds,
            predictor=self.predictor,
            optimizer=torch.optim.Adam(self.actor_model.parameters(), **self.opt_options_actor),
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
            prediction_horizon=1
        )

    def initialize_simulator(self):
        self.simulator = Simulator(
            sys_type="diff_eqn",
            compute_closed_loop_rhs=self.system.compute_closed_loop_rhs,
            sys_out=self.system.out,
            state_init=self.system.state_init,
            disturb_init=[],
            action_init=self.action_init,
            time_start=self.time_start,
            time_final=self.time_final,
            sampling_time=self.sampling_time,
            max_step=self.sampling_time / 10,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dynamic_controller=self.is_dynamic_controller,
        )


    def initialize_scenario(self):

        self.scenario = EpisodicScenarioMarsLander(
            system=self.system,
            simulator=self.simulator,
            controller=self.controller,
            actor=self.actor,
            critic=self.critic,
            logger=self.logger,
            datafiles=self.datafiles,
            time_final=self.time_final,
            running_objective=self.running_objective,
            no_print=self.no_print,
            is_log=self.is_log,
            is_playback=self.is_playback,
            N_episodes=self.N_episodes,
            N_iterations=self.N_iterations,
            state_init=self.system.state_init,
            action_init=self.action_init,
            learning_rate=self.learning_rate_actor,
        )

    def initialize_visualizer(self):
        """
        TODO: change to real parameters
        """
        self.animator = AnimatorMarsLander(
            system=self.system,
            scenario=self.scenario,
        )

    def main_loop_visual(self):
        """
            TODO: implement
            STATUS: in progress
        """
        anm = animation.FuncAnimation(
            self.animator.fig_sim,
            self.animator.animate,
            init_func=self.animator.init_anim,
            blit=False,
            interval=self.sampling_time / 1e6,
            repeat=False,
            # frames=frames,
        )

        self.animator.get_anm(anm)

        cId = self.animator.fig_sim.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )

        anm.running = True

        self.animator.fig_sim.tight_layout()

        plt.show()

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_predictor()
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.initialize_scenario()
        if not self.no_visual and not self.save_trajectory:
            self.initialize_visualizer()
            self.main_loop_visual()
        else:
            self.scenario.run()
