from rcognita_framework.rcognita.scenarios import EpisodicScenario
from rcognita_framework.rcognita.optimizers import TorchOptimizer


def get_mean(array):
    return sum(array)/len(array)

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
