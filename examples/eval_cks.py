from pickletools import optimize
from nn_pruning.patch_coordinator import (
    
    ModelPatchingCoordinator,
    SparseTrainingArguments,
)
import torch
import argparse
from transformers import AutoModelForSequenceClassification


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cks', required=True)
    parser.add_argument('--tasks', required=True)
    args = parser.parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.cks)
    task_name = args.tasks
    
    