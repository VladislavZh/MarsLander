from rcognita_framework.rcognita.scenarios import EpisodicScenario


class EpisodicScenarioMarsLander(EpisodicScenario):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init(*args, **kwargs)
        # Question: how do we update actor correctly? Mb we need online scenario?
