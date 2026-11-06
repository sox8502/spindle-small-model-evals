from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "auto1105r2-qwen35-4b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-auto1105r2"
    platform["secrets"]["api"] = "spindle-auto1105r2-api"



config = Config()
