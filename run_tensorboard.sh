#!/bin/bash
# http://priss:5001

docker run -it -d --rm --name "tensorboard" \
  -p 5000:8888 -p 5001:6006\
  -v $(pwd)//rl-baselines-zoo:/rl-baselines-zoo \
  keshaviyengar/ctm2-stable-baselines:latest \
  bash -c "tensorboard --host 0.0.0.0 --logdir='rl-baselines-zoo/logs/'"

