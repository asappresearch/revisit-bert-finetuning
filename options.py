import argparse
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from transformers import glue_processors as processors

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--save_best", action="store_true", help="Set this flag if you want to save the early stop model.",
    )
    parser.add_argument(
        "--save_last", action="store_true", help="Set this flag if you want to save the last model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=0,
        type=int,
        help="Batch size per GPU/CPU for training to override per_gpu_train_batch_size",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--layerwise_learning_rate_decay", default=1.0, type=float, help="layerwise learning rate decay",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear ratio over total steps.")
    parser.add_argument("--weight_logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--num_loggings", type=int, default=0, help="Total amount of evaluations in training.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    # Added by our paper
    parser.add_argument("--use_bertadam", action="store_true", help="No bias correction")
    parser.add_argument("--use_torch_adamw", action="store_true", help="Use pytorch adamw")
    parser.add_argument(
        "--downsample_trainset", default=-1, type=int, help="down sample training set to this number.",
    )
    parser.add_argument("--resplit_val", default=0, type=int, help="Whether to get the (simulated) test accuracy.")
    parser.add_argument(
        "--reinit_layers",
        type=int,
        default=0,
        help="re-initialize the last N Transformer blocks. reinit_pooler must be turned on.",
    )
    parser.add_argument(
        "--reinit_pooler", action="store_true", help="reinitialize the pooler",
    )
    parser.add_argument("--rezero_layers", type=int, default=0, help="re-zero layers")
    parser.add_argument("--mixout", type=float, default=0.0, help="mixout probability (default: 0)")
    parser.add_argument(
        "--prior_weight_decay", action="store_true", help="Weight Decaying toward the bert params",
    )
    parser.add_argument(
        "--test_val_split", action="store_true", help="Split the original development set in half",
    )

    return parser
