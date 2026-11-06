from copy import deepcopy

from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "auto1105-nemotron35-lightning"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-auto1105"
    platform["secrets"]["api"] = "spindle-auto1105-api"



config = Config()
