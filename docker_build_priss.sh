#!/bin/bash
echo "Buildng keshaviyengar/ctm2-stable-baselines docker image locally."
docker build --rm -t keshaviyengar/ctm2-stable-baselines /home/keshav/ctm2-stable-baselines/
echo "Pushing keshaviyengar/ctm2-stable-baselines to docker hub."
docker push keshaviyengar/ctm2-stable-baselines
echo "Pulling keshaviyengar/ctm2-stable-baselines docker image at priss"
ssh keshav@priss docker pull keshaviyengar/ctm2-stable-baselines
echo "Copying over new rl-baselines-zoo files."
scp -r /home/keshav/ctm2-stable-baselines/rl-baselines-zoo keshav@priss:/home/keshav/ctm2-stable-baselines/
