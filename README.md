# CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries


Submitted to ICML 2025. 



## Installation

Experiments require MuJoCo. 
Please follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Please download the datasets from [LiRE](https://github.com/chwoong/LiRE), and put the dataset in `./data/metaworld_data/<task_name>` and `./data/dmcontrol_data/<task_name>`.

## Run experiments

Train the reward model using CLARIFY, for hammer task with $\epsilon=0.5$:

```bash
python train_contrastive_reward.py  --env hammer --gpu <GPU number> --seed <seed>  \
    --max_feedback 1000 --teacher_eps_skip 0.5 --feed_type "c"
```

Train the reward model using OPRL, for walker-walk task with $\epsilon=0.7$:

```bash
python train_contrastive_reward.py  --env walker-walk --gpu <GPU number> --seed <seed>  \
    --max_feedback 200 --teacher_eps_skip 0.7 --feed_type "d"
```

Train the offline policy based on CLARIFY's reward model, for hammer task with $\epsilon=0.5$::

```bash
python scripts/reward_model_mapping.py
python policy_learning/oprl_policy.py --env hammer --gpu <GPU number> --seed <seed>  \
    --teacher_eps_skip 0.5 --feed_type "c" \
    --reward_model_name_mapping "scripts/reward_model_map_q50.json" \

```

Train the offline policy based on OPRL's reward model, for walker-walk task with $\epsilon=0.7$:

```bash
python scripts/reward_model_mapping.py
python policy_learning/oprl_policy.py --env walker-walk --gpu <GPU number> --seed <seed>  \
    --teacher_eps_skip 0.7 --feed_type "d" \
    --reward_model_name_mapping "scripts/reward_model_map_q50.json" \
```



## Acknowledgement

This repo benefits from [LiRE](https://github.com/chwoong/LiRE), [HIM](https://github.com/frt03/generalized_dt) and [BPref](https://github.com/rll-research/BPref). Thanks for their wonderful work.


