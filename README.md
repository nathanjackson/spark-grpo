# RL on the DGX Spark

There are several different libraries for reinforcement learning, but I haven't
had really any success with adapting them to a custom environment or tasks on
the DGX Spark. So, I've written my own implementation of GRPO that's intended
to "just work" and be hackable for your task.

The current implementation trains a small model, IBM's Granite 4.0 350m, to
play a simple Blackjack game that I vibecoded.

## Getting Started

All instructions assume you're on an up to date Ubuntu 24.04, running on a DGX
Spark.

1. Create a virtual environment, activate, update pip.

    python3 -m venv env
    . ./env/bin/activate
    pip install -U pip

2. Install dependencies.

    pip install -r requirements.txt

3. Training - While running the training script, you should eventually see one
   of the evals come back with a win rate of ~41%, which is the best I was able
   to get.

    python3 train.py

5. Test - There is also a test script that you can use to compare the win rates
   of the original model against your checkpoint.

    # Original model: ibm-granite/granite-4.0-350m
    python3 test_play.py --games 1000

    # Your checkpoint
    python3 test_play.py --model-path <path to checkpoint folder> --games 1000

