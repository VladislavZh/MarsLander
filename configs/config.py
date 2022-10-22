from rcognita_framework.pipelines.config_blueprints import AbstractConfig


class ConfigMarsLander(AbstractConfig):
    def __init__(self):
        self.config_name = "mars-lander"

    def argument_parser(self):
        pass

    def pre_processing(self):
        pass
