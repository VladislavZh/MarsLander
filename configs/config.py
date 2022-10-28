from rcognita_framework.pipelines.config_blueprints import AbstractConfig, RcognitaArgParser
from collections import namedtuple
import numpy as np


class ConfigMarsLander(AbstractConfig):
    def __init__(self):
        self.config_name = "mars-lander"

    def argument_parser(self):
        description = (
            "Agent-environment pipeline: Mars lander on a random surface."
        )

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--is_playback", action="store_true", help="Flag to playback.",
        )

        parser.add_argument(
            "--t1_critic",
            type=float,
            metavar="time_final_critic",
            dest="time_final_critic",
            default=100.0,
            help="Final time of critic episode.",
        )
        parser.add_argument(
            "--N_episodes",
            type=int,
            default=4,
            help="Number of episodes in one actor iteration",
        )
        parser.add_argument(
            "--N_iterations",
            type=int,
            default=10,
            help="Number of iterations in episodical actor scenario",
        )

        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=1.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time sampling_time.",
        )

        parser.add_argument(
            "--learning_rate_actor",
            type=float,
            default=0.001,
            help="Size of NN actor learning rate.",
        )

        parser.add_argument(
            "--learning_rate_critic",
            type=float,
            default=0.001,
            help="Size of NN critic learning rate.",
        )

        parser.add_argument(
            "--weight_decay_actor",
            type=float,
            default=0.00001,
            help="Size of NN actor learning weight decay.",
        )

        parser.add_argument(
            "--weight_decay_critic",
            type=float,
            default=0.00001,
            help="Size of NN critic learning weight decay.",
        )

        parser.add_argument(
            "--speedup", type=int, default=20, help="Animation speed up",
        )

        parser.add_argument(
            "--data_buffer_size",
            type=int,
            default=100,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )

        parser.add_argument(
            "--discount_factor", type=float, default=1.0, help="Discount factor."
        )

        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times sampling_time seconds.",
        )

        parser.add_argument(
            "--fuel_consumption_coeff",
            type=float,
            default=0.0,
            help="Fuel consumption cost multiplier.",
        )

        parser.add_argument(
            "--angle_constraint_coeff",
            type=float,
            default=0.0,
            help="Angle cost multiplier.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 5
        self.dim_input = 2
        self.dim_output = 28
        self.dim_disturb = 0

        self.pred_step_size = self.sampling_time * self.pred_step_size_multiplier
        self.critic_period = self.sampling_time * self.critic_period_multiplier

        self.is_disturb = 0

        self.is_dynamic_controller = 0

        self.time_start = 0

        self.action_init = np.zeros(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # Control constraints
        self.action_bounds = np.array([[0, 5],[-1,1]])
        self.prediction_horizon = 1

        # System parameters
        self.g = 3.7
        self.observation_target = []

        self.control_mode = "Actor-Critic"
