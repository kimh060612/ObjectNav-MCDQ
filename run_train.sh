#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python3 -m habitat_baselines.run --exp-config habitat_baselines/config/objectnav/ppo_uncertain_objectnav.yaml --run-type train