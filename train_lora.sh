#!/bin/env bash

source ./backend/sd_scripts/venv/bin/activate
./train_lora.py /home/timmy/lora-training/jobs/queue/ "$@"
