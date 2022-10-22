from rcoginta_framework.rcoginta.systems import System


class SysMarsLander(System):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.name = "mars-lander"

    def _compute_state_dynamics(self, time, state, action, disturb=[]):

        g = self.pars[0]

        Dstate = None# code

        return Dstate

    def out(self, state, time=None, action=None):

        observation = None # code

        return observation

    def reset(self):
        pass
