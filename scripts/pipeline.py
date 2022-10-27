import sys
from pathlib import Path

from rcognita_framework.rcognita import optimizers

sys.path.append(Path("../rcognita_framework").resolve())

from rcognita_framework.pipelines.pipeline_blueprints import PipelineWithDefaults, AbstractPipeline
from rcognita_framework.rcognita.models import ModelGaussianConditional

from rcognita_framework.rcognita.optimizers import TorchOptimizer
from configs import ConfigMarsLander

from .models import ModelActorMarsLander, ModelCriticMarsLander, ModelRunningObjectiveMarsLander
from .actor import ActorMarsLander, ActorMarsLanderAC, ActorProbabilisticEpisodicACMars
from .animator import AnimatorMarsLander
from .critic import CriticMarsLander, CriticMarsLanderAC
from .scenario import EpisodicScenarioMarsLander, EpisodicScenarioDQN
# from .simulator import SimulatorMarsLander
from .system import SysMarsLander

import matplotlib.animation as animation
from rcognita.utilities import on_key_press
import matplotlib.pyplot as plt
import numpy as np


# class PipelineMarsLander(PipelineWithDefaults):
class PipelineMarsLanderAC(AbstractPipeline):
    config = ConfigMarsLander

    def initialize_logger(self):
        self.logger = None

    def initialize_controller(self):
        self.controller = None

    def initialize_predictor(self):
        self.predictor = None

    def initialize_simulator(self):
        self.simulator = None

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
        """
            TODO: fix
        """
        self.critic_model = ModelCriticMarsLander(
                                self.dim_output,
                                # ... # model params
                            )
        self.actor_model  = ModelActorMarsLander(
                                self.dim_output,
                                self.dim_input,
                                # ... # model params
                            )

        self.model_running_objective = ModelRunningObjectiveMarsLander(
                                fuel_consumption_coeff=self.fuel_consumption_coeff,
                                angle_constraint_coeff=self.angle_constraint_coeff
                            )


    def initialize_optimizers(self):
        opt_options_actor = {
            "lr": self.actor_lr,
            "weight_decay": self.actor_weight_decay
        }
        opt_options_critic = {
            "lr": self.critic_lr,
            "weight_decay": self.critic_weight_decay
        }

        self.actor_optimizer = optimizers.TorchOptimizer(
            opt_options_actor, iterations=self.actor_iterations
        )
        self.critic_optimizer = optimizers.TorchOptimizer(
            opt_options_critic, iterations=self.critic_iterations
        )

    def initialize_actor_critic(self):
        """
            TODO: fix
        """

        self.critic = CriticMarsLanderAC(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size, # do we need it?
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            sampling_time=self.sampling_time,
        )

        self.actor = ActorMarsLanderAC(
            #self.prediction_horizon,
            self.dim_input,
            self.dim_output,
            self.control_mode, # what is it?
            self.action_bounds, # maybe better to do it implicitly?
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
            action_bounds=[[0, -1], [5,1]]
        )

    # def initialize_simulator(self):
    #     self.simulator = SimulatorMarsLander(
    #         sys_type="diff_eqn",
    #         compute_closed_loop_rhs=self.system.compute_closed_loop_rhs,
    #         sys_out=self.system.out,
    #         state_init=self.state_init,
    #         disturb_init=[],
    #         action_init=self.action_init,
    #         time_start=self.time_start,
    #         time_final=self.time_final,
    #         sampling_time=self.sampling_time,
    #         max_step=self.sampling_time / 10,
    #         first_step=1e-6,
    #         atol=self.atol,
    #         rtol=self.rtol,
    #         is_disturb=self.is_disturb,
    #         is_dynamic_controller=self.is_dynamic_controller,
    #     )


    def initialize_scenario(self):
        self.scenario = EpisodicScenarioMarsLander(
            self.N_episodes,
            self.N_iterations,
            self.system,
            self.simulator,
            self.controller,
            self.actor,
            self.critic,
            self.logger,
            self.datafiles,
            self.time_final,
            self.running_objective,
            no_print=self.no_print,
            is_log=self.is_log,
        )

    def initialize_visualizer(self):
        """
        TODO: change to real parameters
        """
        # angle_init = 0
        self.animator = AnimatorMarsLander(
            initial_coords=[7000, 3000, 0], #self.simulator.state_full_init,
            landscape=self.system.landscape,
            xs=np.array([self.system.landscape[:, 0].max() - i for i in range(10, 100)]), #self.simulator.state[0],
            ys=np.array([self.system.landscape[:, 1].max() - i*40 for i in range(10, 100)]), #self.simulator.state[1],
            angles=np.linspace(0, 45, 9), #self.simulator.state[2],
        )

    def main_loop_visual(self):
        """
            TODO: implement
            STATUS: in progress
        """
        # frames = np.arange(100)
        anm = animation.FuncAnimation(
            self.animator.fig_sim,
            self.animator.animate,
            init_func=self.animator.init_anim,
            blit=False,
            interval=self.sampling_time / 1e6,
            repeat=False,
            # frames=frames,
        )
        # anm = animation.FuncAnimation(
        #     self.visualizer.fig_sim,
        #     self.visualizer.animate,
        #     init_func=self.visualizer.init_anim,
        #     blit=False,
        #     interval=self.sampling_time / 1e6,
        #     repeat=False,
        # )

        self.animator.get_anm(anm)
        # self.visualizer.get_anm(anm)

        cId = self.animator.fig_sim.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )
        # cId = self.visualizer.fig_sim.canvas.mpl_connect(
        #     "key_press_event", lambda event: on_key_press(event, anm)
        # )

        anm.running = True

        self.animator.fig_sim.tight_layout()
        # self.visualizer.fig_sim.tight_layout()

        plt.show()

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_predictor()
        # self.initialize_safe_controller()
        self.initialize_models()
        # self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.initialize_scenario()
        # if not self.no_visual and not self.save_trajectory:
        self.initialize_visualizer()
        self.main_loop_visual()
        # else:
        #     self.scenario.run()



class PipelineMarsLanderDQN(AbstractPipeline):
    config = ConfigMarsLander

    def initialize_models(self):
        self.actor_model =ModelGaussianConditional(
            expectation_function=self.safe_controller,
            arg_condition=self.observation_init,
            weights=self.initial_weights,
        )
        self.critic_model = ModelCriticMarsLander(
                                self.dim_output,
                                self.dim_input,
                                # ... # model params
                            )
        self.model_running_objective = ModelRunningObjectiveMarsLander(
                                fuel_consumption_coeff=self.fuel_consumption_coeff,
                                angle_constraint_coeff=self.angle_constraint_coeff
                            )

    def initialize_actor_critic(self):
        self.critic = CriticMarsLanderAC(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            data_buffer_size=self.data_buffer_size, # do we need it?
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            sampling_time=self.sampling_time,
        )
        self.actor = ActorProbabilisticEpisodicACMars(
            self.prediction_horizon,
            self.dim_input,
            self.dim_output,
            self.control_mode,
            self.action_bounds,
            action_init=self.action_init,
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
        )

    def initialize_scenario(self):
        self.scenario = EpisodicScenarioDQN(
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
            state_init=self.state_init,
            action_init=self.action_init,
            learning_rate=self.learning_rate
        )

    def execute_pipeline(self, **kwargs):
        """
        Full execution routine
        """
        np.random.seed(42)
        super().execute_pipeline(**kwargs)

if __name__ == "__main__":
    pipeline = PipelineMarsLanderDQN()
    pipeline.execute_pipeline()
    
