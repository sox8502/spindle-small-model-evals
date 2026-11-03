from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "rerun1103-qwen35-4b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-rerun1103"
    platform["secrets"]["api"] = "spindle-rerun1103-api"


config = Config()
