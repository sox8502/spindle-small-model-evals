from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "evals-qwen35-4b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-evals"
    platform["secrets"]["api"] = "spindle-evals-api"


config = Config()
