from pickletools import optimize
from nn_pruning.patch_coordinator import (
    
    ModelPatchingCoordinator,
    SparseTrainingArguments,
)
import torch
from transformers import AutoModelForSequenceClassification
sparse_args = SparseTrainingArguments()



hyperparams = {

    "dense_pruning_method": "topK:1d_alt",

    "attention_pruning_method": "topK",

    "initial_threshold": 1.0,

    "final_threshold": 0.1,

    "initial_warmup": 1,

    "final_warmup": 3,

    "attention_block_rows": 32,

    "attention_block_cols": 32,

    "attention_output_with_dense": 0,

    "distil_alpha_ce": 0,

    "distil_alpha_teacher": 1.0,

    "distil_teacher_name_or_path": "aloxatel/bert-base-mnli",

    "distil_temperature": 1.0

}



for k, v in hyperparams.items():

    if hasattr(sparse_args, k):

        setattr(sparse_args, k, v)

    else:

        print(f"sparse_args does not have argument {k}")
mpc = ModelPatchingCoordinator(

    sparse_args=sparse_args,

    device=torch.device('cuda:0'),

    cache_dir="checkpoints",

    logit_names=["logits"],

    teacher_constructor=None,

    model_name_or_path="aloxatel/bert-base-mnli",

)
model = AutoModelForSequenceClassification.from_pretrained('mnli_topk')
mpc.patch_model(model)
model.load_state_dict(torch.load('mnli_topk/pytorch_model.bin'))
mpc.compile_model(model)
import ipdb; ipdb.set_trace()