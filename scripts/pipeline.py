from rcognita_framework.pipelines.pipeline_blueprints import PipelineWithDefaults
from rcognita_framework.rcognita.optimizers import TorchOptimizer
from configs import ConfigMarsLander

from .models import ModelActorMarsLander, ModelCriticMarsLander, ModelRunningObjectiveMarsLander
from .actor import ActorMarsLander
from .critic import CriticMarsLander
from .scenario import EpisodicScenarioMarsLander
from .system import SysMarsLander



class PipelineMarsLander(PipelineWithDefaults):
    config = ConfigMarsLander

    def initialize_system(self):
        self.system = SysMarsLander()

    def initialize_models(self):
        """
            TODO: fix
        """
        self.critic_model = ModelCriticMarsLander(
                                self.dim_observation,
                                ... # model params
                            )
        self.actor_model  = ModelActorMarsLander(
                                self.dim_observation,
                                self.dim_action,
                                ... # model params
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

        self.critic = CriticMarsLander(
            dim_input=self.dim_input, # not clear
            dim_output=self.dim_output, # not clear
            data_buffer_size=self.data_buffer_size, # do we need it?
            running_objective=self.running_objective,
            discount_factor=self.discount_factor,
            optimizer=self.critic_optimizer,
            model=self.critic_model,
            sampling_time=self.sampling_time,
        )

        self.actor = ActorMarsLander(
            self.dim_input,
            self.dim_output,
            self.control_mode, # what is it?
            self.action_bounds, # maybe better to do it implicitly?
            predictor=self.predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_objective=self.running_objective,
            model=self.actor_model,
        )

    def initialize_scenario(self):
        self.scenario = EpisodicScenarioMarsLander(
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

    def main_loop_visual(self):
        """
            TODO: implement
        """
        raise NotImplementedError
