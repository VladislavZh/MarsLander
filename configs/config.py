from rcognita_framework.pipelines.config_blueprints import AbstractConfig, RcognitaArgParser
from collections import namedtuple
import numpy as np

class ConfigMarsLander(AbstractConfig):
    def __init__(self):
        self.config_name = "mars-lander"

    def argument_parser(self):
        # print("argument_parser")
        # parser = RcognitaArgParser(description="mars-lander")

        # parser.add_argument(
        #     "--dim_state",
        #     type=int,
        #     default=2,
        #     help="dim_state",
        # )

        # parser.add_argument(
        #     "--dim_input",
        #     type=int,
        #     default=2,
        #     help="dim_input",
        # )

        # parser.add_argument(
        #     "--dim_output",
        #     type=int,
        #     default=1,
        #     help="dim_output",
        # )

        # parser.add_argument(
        #     "--dim_disturb",
        #     type=int,
        #     default=1,
        #     help="dim_disturb",
        # )

        # parser.add_argument(
        #     "--g",
        #     type=float,
        #     default=9.8,
        #     help="g",
        # )

        # parser.add_argument(
        #     "--g",
        #     type=float,
        #     default=9.8,
        #     help="g",
        # )

        # parser.add_argument(
        #     "--g",
        #     type=float,
        #     default=9.8,
        #     help="g",
        # )

        # return parser.parse_args()
        return dict(sys_type='diff_eqn',
                    dim_state=5,
                    dim_input=2,
                    dim_output=25,
                    dim_disturb=0,
                    g=3.7,
                    is_dynamic_controller=False,
                    is_disturb=False,
                    #prediction_horizon=10,
                    fuel_consumption_coeff=0.95,
                    angle_constraint_coeff=np.pi / 2,
                    actor_lr=1e-3,
                    actor_weight_decay=1e-5,
                    critic_lr=1e-3,
                    critic_weight_decay=1e-5,
                    actor_iterations=10,
                    critic_iterations=10,
                    data_buffer_size=4,
                    discount_factor=0.99,
                    sampling_time=1,
                    control_mode="Actor-Critic",
                    action_bounds={'lb':[0, -1],'ub':[5,1]},
                    action_init=[],
                    predictor=[],
                    datafiles=[""],
                    time_start=0,
                    time_final=200,
                    no_print=True,
                    is_log=False,
                    N_episodes=1,
                    N_iterations=10,
                    no_visual=True,
                    save_trajectory=True,
                    critic_period=1,
                    atol=1e-5,
                    rtol=1e-3

                    # action_min=0,
                    # action_max=1e10,
                )


        #         action_bounds=[],

        # predictor=[],
        # optimizer=None,
        # critic=[],
        # running_objective=[],
        # model=None,
        # discount_factor=1,

        #     self.critic = CriticMarsLander(
        #     optimizer=self.critic_optimizer,
        #     model=self.critic_model,
        # )

        # self.actor = ActorMarsLander(
        #     self.dim_input,
        #     self.dim_output,
        #     self.control_mode, # what is it?
        #     self.action_bounds, # maybe better to do it implicitly?
        #     predictor=self.predictor,
        #     optimizer=self.actor_optimizer,
        #     critic=self.critic,
        #     running_objective=self.running_objective,
        #     model=self.actor_model,
        # )

    def pre_processing(self):
        pass
