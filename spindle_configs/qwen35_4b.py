from copy import deepcopy

from spindle.configs.qwen35_4b_lora_32k import Config as Parent


class Config(Parent):
    name = "ctlg1103-qwen35-4b"
    # Own frontend and API secret so this project never collides with another Spindle deployment.
    platform = deepcopy(Parent.platform)
    platform["frontend"] = "spindle-ctlg1103"
    platform["secrets"]["api"] = "spindle-ctlg1103-api"

    inference_min_replicas = 1
    inference_max_replicas = 1
    pool_idle_timeout_s = 900
    session_idle_timeout_s = 7200


config = Config()
