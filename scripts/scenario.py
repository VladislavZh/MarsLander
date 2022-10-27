from rcognita_framework.rcognita.scenarios import EpisodicScenario
from rcognita_framework.rcognita.optimizers import TorchOptimizer


def get_mean(array):
    return sum(array)/len(array)

class EpisodicScenarioCriticLearnMarsLander(EpisodicScenario):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as np

        angle_inits = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, self.N_iterations)
        angular_velocity_inits = np.random.uniform(
            -np.pi / 2.0, np.pi / 2.0, self.N_iterations
        )

        w1s = np.random.uniform(0, 15, self.N_iterations)
        w2s = np.random.uniform(0, 15, self.N_iterations)
        w3s = np.random.uniform(0, 15, self.N_iterations)

        self.state_inits = np.vstack((angle_inits, angular_velocity_inits)).T
        self.actor_model_weights = np.vstack((w1s, w2s, w3s)).T

        self.action_inits = np.random.uniform(-25.0, 25.0, self.N_iterations)
        self.critic_loss_values = []

    def init_conditions_update(self):
        self.simulator.state_full_init = self.state_init = self.state_inits[
            self.iteration_counter, :
        ]
        self.action_init = self.action_inits[self.iteration_counter]
        self.actor.model.weights = self.actor_model_weights[self.iteration_counter, :]

    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.outcome = 0
        self.init_conditions_update()
        self.action = self.action_init
        self.system.reset()
        self.actor.reset()
        self.critic.reset()
        self.controller.reset(time_start=0)
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0

    def run(self):
        self.step_counter = 0
        self.one_episode_steps_numbers = [0]
        skipped_steps = 43
        for _ in range(self.N_iterations):
            for _ in range(self.N_episodes):
                while self.sim_status not in [
                    "episode_ended",
                    "simulation_ended",
                    "iteration_ended",
                ]:
                    self.sim_status = self.step()
                    self.step_counter += 1
                    if self.step_counter > skipped_steps:
                        self.critic_loss_values.append(self.critic.current_critic_loss)

                self.one_episode_steps_numbers.append(
                    self.one_episode_steps_numbers[-1]
                    + self.step_counter
                    - skipped_steps
                )
                self.step_counter = 0
                if self.sim_status != "simulation_ended":
                    self.reload_pipeline()
        if self.is_playback:
            if len(self.episode_tables) > 1:
                self.episode_tables = rc.vstack(self.episode_tables)
            else:
                self.episode_tables = rc.array(self.episode_tables[0])

        self.plot_critic_learn_results()

    def plot_critic_learn_results(self):
        figure = plt.figure(figsize=(9, 9))
        ax_critic = figure.add_subplot(111)
        ax_critic.plot(self.critic_loss_values, label="TD")
        [ax_critic.axvline(i, c="r") for i in self.one_episode_steps_numbers]
        plt.legend()
        plt.savefig(
            f"./critic/{self.N_iterations}-iters_{self.time_final}-fintime_{self.critic.data_buffer_size}-dbsize",
            format="png",
        )
        plt.show()


class EpisodicScenarioMarsLander(EpisodicScenario):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Question: how do we update actor correctly? Mb we need online scenario?

class EpisodicScenarioMarsLanderAC(EpisodicScenario):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init(*args, **kwargs)
        # Question: how do we update actor correctly? Mb we need online scenario?
        self.actor_optimizer = TorchOptimizer({"lr": 0.01})
        self.critic_optimizer = TorchOptimizer({"lr": 0.01})
        self.squared_TD_sums_of_episodes = []
        self.square_TD_means = []

    def reset_episode(self):
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.squared_TD_sums_of_episodes.append(self.critic.objective())
        #self.actor_loss_sums_of_episodes.append(self.actor.objective())
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().reset_episode()

    def iteration_update(self):
        """
        Proposition: create a lambda function that evaluates mean
        to pass this function as an objective into your optimizer.
        Keep in mind that you must not detach your tensors with inplace somehow.
        If you call, for example, torch.tensor(tens), it will detach tens automatically
        """
        mean_sum_of_squared_TD = get_mean(self.squared_TD_sums_of_episodes) #just for visualization purposes
        #mean_sum_of_squared_TD = get_mean(self.squared_TD_sums_of_episodes) #just for visualization purposes
        self.square_TD_means.append(mean_sum_of_squared_TD.detach().numpy()) #just for visualization purposes
        self.actor_loss_sums_of_episodes = self.actor.losses_iter
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.critic_optimizer.optimize(
            objective=get_mean,
            model=self.critic.model,
            model_input=self.squared_TD_sums_of_episodes,
        )
        self.actor_optimizer.optimize(
            objective=get_mean,
            model=self.actor.model,
            model_input=self.actor_loss_sums_of_episodes,
        )
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().iteration_update()

    def reset_iteration(self):
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.squared_TD_sums_of_episodes = []
        self.actor_loss_sums_of_episodes = []
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().reset_iteration()

####################################################
class EpisodicScenarioDQN(EpisodicScenario):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_optimizer = TorchOptimizer({"lr": 0.01})
        self.squared_TD_sums_of_episodes = []
        self.square_TD_means = []

    def reset_episode(self):
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.squared_TD_sums_of_episodes.append(self.critic.objective())
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().reset_episode()

    def iteration_update(self):
        """
        Proposition: create a lambda function that evaluates mean
        to pass this function as an objective into your optimizer.
        Keep in mind that you must not detach your tensors with inplace somehow.
        If you call, for example, torch.tensor(tens), it will detach tens automatically
        """
        mean_sum_of_squared_TD = get_mean(self.squared_TD_sums_of_episodes) #just for visualization purposes
        self.square_TD_means.append(mean_sum_of_squared_TD.detach().numpy()) #just for visualization purposes
        print(self.square_TD_means[-1])
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.critic_optimizer.optimize(
            objective=get_mean,
            model=self.critic.model,
            model_input=self.squared_TD_sums_of_episodes,
        )
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().iteration_update()

    def reset_iteration(self):
        #############################################
        # YOUR CODE BELOW
        #############################################
        self.squared_TD_sums_of_episodes = []
        #############################################
        # YOUR CODE ABOVE
        #############################################
        super().reset_iteration()
