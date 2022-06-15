# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2022-05-08 23:08:58
# @Last Modified by:   gunjianpan
# @Last Modified time: 2022-05-29 00:08:34

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import joblib
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments,
    ViTConfig,
    ViTForImageClassification,
    ViTFeatureExtractor,
    get_cosine_schedule_with_warmup,
)
from nn_pruning.model_structure import struct_from_config
from nn_pruning.inference_model_patcher import optimize_model
from nn_pruning.patch_coordinator import (
    ModelPatchingCoordinator,
    SparseTrainingArguments,
)
from nn_pruning.sparse_trainer import SparseTrainer

# from dataloader_wrapper import ImageZipDatasetWrapper
from head_pruner import str2att_mask_list
from data.dataset import ImageFolder


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def build_dataset(zip_dir: str, img_prefix: str, info_path: str, transform):
    return ImageFolder(
        zip_dir,
        ann_file=info_path,
        img_prefix=img_prefix,
        transform=transform,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_schedule_with_warmup(optimizer, last_epoch=-1):
    def lr_lambda(current_step):
        return 0.1 ** (current_step // 30)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class PruningTrainer(SparseTrainer, Trainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an
        error when run without distillation
        """
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = loss.mean()
        self.metrics["ce_loss"] += float(loss)

        loss, distil_loss = self.patch_coordinator.distil_loss_combine(
            loss, inputs, outputs
        )
        loss = loss.mean()
        self.metrics["distil_loss"] += float(distil_loss)

        regu_loss, lamb, info = self.patch_coordinator.regularization_loss(model)

        for kind, values in info.items():
            if kind == "total":
                suffix = ""
            else:
                suffix = "_" + kind

            for k, v in values.items():
                self.metrics[k + suffix] += float(v)
        loss = loss + regu_loss * lamb
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, model):
        args = self.args

        optimizer_grouped_parameters = self.patch_coordinator.create_optimizer_groups(
            model, self.args, self.sparse_args
        )

        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            args.learning_rate,
            momentum=0.9,
        )
        return optimizer

    def create_scheduler(self, num_training_steps: int):
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.args.warmup_steps,
            num_training_steps,
        )
        return scheduler


@dataclass
class PruningArguments:
    distil_teacher_name_or_path: str = field(
        default=None, metadata={"help": "directory containing model to be pruned"}
    )
    final_threshold: float = field(default=0.5)
    block_size_row: int = field(default=1)
    block_size_col: int = field(default=1)
    head_mask_str: str = field(default="empty", metadata={"help": "head mask str"})
    shared_weight: str = field(default="")


def get_mask_scores(model):
    model_struct = struct_from_config(model.config_class)
    key_lists = [
        model_struct.LAYER_PATTERNS.get(jj, jj)
        for jj in ["query", "key", "value", "att_dense", "att_weight_mask"]
    ]
    res = {}
    for k, v in model.named_modules():
        if any(k.endswith(ii) for ii in key_lists):
            res[k] = v.weight.data.detach().cpu().numpy() == 0
            # if "mask_module" not in v._modules:
            #     continue
            # res[k] = v.mask_module.context_modules[0].mask_scores.data.detach().cpu().numpy()
    return res


def get_sparity(model):
    config = model.config
    stats = {
        "total": config.num_hidden_layers
        * (
            4 * config.hidden_size * config.hidden_size
            + 2 * config.hidden_size * config.intermediate_size
        ),
        "nnz": 0,
        "linear_attention_total": config.num_hidden_layers
        * (4 * config.hidden_size * config.hidden_size),
        "linear_attention_nnz": 0,
    }
    model_structure = struct_from_config(model.config_class)
    for name, parameter in model.named_parameters():
        is_attention = model_structure.is_attention(name)
        is_ffn = model_structure.is_ffn(name)
        is_layernorm = model_structure.is_layernorm(name)
        is_linear_layer_weight = (
            (is_attention or is_ffn) and name.endswith(".weight") and not is_layernorm
        )

        nnz = int((parameter != 0).sum())
        stats["nnz"] += nnz

        if is_linear_layer_weight:
            if is_attention:
                stats["linear_attention_nnz"] += nnz

    stats["sparity"] = stats["nnz"] / stats["total"]
    stats["sparity_attention"] = (
        stats["linear_attention_nnz"] / stats["linear_attention_total"]
    )
    return stats


def set_attention_weight_mask(model):
    T = 256
    for k, v in model.named_modules():
        if k.endswith("att_weight_mask"):
            v.weight.data = torch.ones(T, T).to(model.device)
            v.weight.requires_grad = False
            v.bias.data = torch.zeros(T).to(model.device)
            v.bias.requires_grad = False
    return model


def set_mask_scores(trainer, threshold: float):
    path = f"results/L1_{threshold:.3f}_1x1_single.pkl"
    model = trainer.model
    mask_scores = joblib.load(path)["mask_scores"]
    for k, v in model.named_modules():
        if k in mask_scores:
            M = mask_scores[k]
            copy_param = torch.nn.Parameter()
            copy_param.data = torch.tensor(M).float().to(model.device)
            # copy_param.requires_grad = False
            v.mask_module.context_modules[0].mask_scores = copy_param
            v.mask_module.context_modules[0].mask_scores.requires_grad = False
            print(v)
    trainer.model = model


if __name__ == "__main__":
    parser = HfArgumentParser((PruningArguments, ModelArguments, TrainingArguments))
    pruning_args, model_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    config = ViTConfig.from_pretrained(model_args.model_name_or_path)
    # config.intermediate_size //= 2
    model = ViTForImageClassification.from_pretrained(
        model_args.model_name_or_path, config=config
    )
    model = optimize_model(model, "dense")

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        model_args.model_name_or_path
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(
        mean=feature_extractor.image_mean, std=feature_extractor.image_std
    )
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB"))
            for pil_img in example_batch["image"]
        ]
        return example_batch

    train_dataset = build_dataset(
        "/tmp/imagenet/", "train.zip@/", "train_wo_dev_map.txt", _train_transforms
    )
    valid_dataset = build_dataset(
        "/tmp/imagenet/", "train.zip@/", "dev_map.txt", _val_transforms
    )
    test_dataset = build_dataset(
        "/tmp/imagenet/", "val.zip@/", "val_map.txt", _val_transforms
    )

    dataset = {
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset,
    }

    heads = str2att_mask_list(pruning_args.head_mask_str)
    model.prune_heads({layer_id: head_ids for layer_id, head_ids in enumerate(heads)})
    logger.info("Done loading model {}".format(heads))

    logger.info("Initializing sparse training arguments")
    sparse_args = SparseTrainingArguments()

    hyperparams = {
        "dense_pruning_method": "topK:1d_alt",
        "attention_pruning_method": "disabled",
        "initial_threshold": 1.0,
        "final_threshold": pruning_args.final_threshold,
        "initial_warmup": 1,
        "final_warmup": 3,
        "attention_block_rows": pruning_args.block_size_row,
        "attention_block_cols": pruning_args.block_size_col,
        "attention_output_with_dense": 0,
        "distil_alpha_ce": 0,
        "distil_alpha_teacher": 1.0,
        "distil_teacher_name_or_path": pruning_args.distil_teacher_name_or_path,
        "distil_temperature": 1.0,
        "shared_weight": pruning_args.shared_weight,
    }
    for k, v in hyperparams.items():
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k}")
    logger.info("Done initializing sparse arguments")

    logger.info("Initializing model patch coordinator")
    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args,
        device=training_args.device,
        cache_dir="checkpoints",
        logit_names=["logits"],
        teacher_constructor=ViTForImageClassification,
        model_name_or_path=pruning_args.distil_teacher_name_or_path,
    )
    mpc.patch_model(model)
    logger.info("Done initializing mpc and patching model")
    logger.info("Using {} gpu (s)".format(training_args.n_gpu))

    trainer = PruningTrainer(
        sparse_args=sparse_args,
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )
    trainer.set_patch_coordinator(mpc)
    # set_mask_scores(trainer, pruning_args.final_threshold)
    trainer.train()
    if training_args.local_rank in [0, -1]:
        mask_scores = get_mask_scores(trainer.model)
        # joblib.dump(trainer.model, "model.pkl")
        mpc.compile_model(trainer.model)
        output_model_path = f"fluency_final_{pruning_args.final_threshold:.3f}_{pruning_args.block_size_row}x{pruning_args.block_size_col}"
        trainer.save_model(output_model_path)

        params = {
            "block_size_row": pruning_args.block_size_row,
            "block_size_col": pruning_args.block_size_col,
            "final_threshold": pruning_args.final_threshold,
            "shared_weight": pruning_args.shared_weight,
            "sparity": get_sparity(trainer.model),
        }
        joblib.dump(
            {"params": params, "mask_scores": mask_scores},
            f"results/{pruning_args.final_threshold:.3f}_{pruning_args.block_size_row}_{pruning_args.block_size_col}_{pruning_args.shared_weight}_{training_args.seed}.pkl",
        )

        prune_hugging_face_model = optimize_model(trainer.model, "dense")
        compression_ratio = (
            prune_hugging_face_model.num_parameters() / model.num_parameters()
        )
        logger.info("percentage of remained weights: {}".format(compression_ratio))
