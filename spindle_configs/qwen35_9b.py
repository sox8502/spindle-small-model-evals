from copy import deepcopy

from spindle.configs.qwen35_9b_instruct_lora_16k import Config as Parent


class Config(Parent):
    name = "one1103-qwen35-9b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-one1103"
    platform["secrets"]["api"] = "spindle-one1103-api"

    inference_max_replicas = 1


config = Config()
