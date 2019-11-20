#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line

docker run -it --rm --network host --ipc=host \
 -v /tmp/rosbags:/rosbags \
 -v $(pwd)/rl-baselines-zoo:/root/code/rl-baselines-zoo \
 keshaviyengar/ctm2-stable-baselines:latest bash -c "cd /root/code/rl-baselines-zoo/ && $cmd_line"
