from copy import deepcopy

from spindle.configs.nemotron35_lightning_30b_a3b_lora_64k import Config as Parent


class Config(Parent):
    name = "ctl1103-nemotron35-lightning"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-ctl1103"
    platform["secrets"]["api"] = "spindle-ctl1103-api"

    inference_min_replicas = 1
    inference_max_replicas = 1
    pool_idle_timeout_s = 7200
    session_idle_timeout_s = 7200


config = Config()
