from copy import deepcopy

from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "auto1105-qwen35-9b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-auto1105"
    platform["secrets"]["api"] = "spindle-auto1105-api"



config = Config()
