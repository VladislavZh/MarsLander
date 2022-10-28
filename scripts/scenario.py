from rcognita_framework.rcognita.scenarios import EpisodicScenario
from rcognita_framework.rcognita.optimizers import TorchOptimizer


def get_mean(array):
    return sum(array)/len(array)

class EpisodicScenarioMarsLander(EpisodicScenario):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.squared_TD_sums_of_episodes = []
        self.square_TD_means = []

    def reload_pipeline(self):
        self.sim_status = 1
        self.time = 0
        self.time_old = 0
        self.outcome = 0
        self.action = self.action_init
        self.system.reset()
        self.state_init = self.system.state_init
        self.actor.reset()
        self.critic.reset()
        self.controller.reset(time_start=0)
        self.simulator.state_full_init = self.system.state_init
        self.simulator.reset()
        self.observation = self.system.out(self.state_init, time=0)
        self.sim_status = 0

    def reset_episode(self):
        self.episode_counter += 1
        self.squared_TD_sums_of_episodes.append(self.critic.objective())
        self.reload_pipeline()

    def iteration_update(self):
        mean_sum_of_squared_TD = self.get_mean(self.squared_TD_sums_of_episodes)
        self.square_TD_means.append(mean_sum_of_squared_TD.detach().numpy())
        print(f"TD Loss = {mean_sum_of_squared_TD.detach().numpy()}")

        self.critic.optimizer.optimize(
            objective=self.get_mean,
            model=self.critic.model,
            model_input=self.squared_TD_sums_of_episodes,
        )

        for p in self.actor.model.parameters():
            p.grad /= len(self.squared_TD_sums_of_episodes)

        self.actor.optimizer.step()

    def reset_iteration(self):
        self.squared_TD_sums_of_episodes = []
        self.actor.optimizer.zero_grad()
        super().reset_iteration()
