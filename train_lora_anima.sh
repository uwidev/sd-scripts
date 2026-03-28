#!/bin/env bash

source ./backend/sd_scripts/venv/bin/activate
./train_lora_anima.py /home/timmy/lora-training/jobs/queue/ "$@"
