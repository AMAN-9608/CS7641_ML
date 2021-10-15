import logging
import os
import sys
import gc
import torch
from datetime import datetime

from src.args import DCNNArguments, DCNNConfig
from src.data import ImageCxDataset
from seqlbtoolkit.IO import set_logging, logging_args

from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def image_cx(args: DCNNArguments):
    set_logging(log_dir=args.log_dir)

    logging_args(args)
    set_seed(args.seed)
    config = DCNNConfig().from_args(args)

    training_dataset = ImageCxDataset().load_file(args.train_dir)
    valid_dataset = training_dataset.pop_random(args.valid_ratio)
    test_dataset = ImageCxDataset().load_file(args.test_dir)


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(DCNNArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        chmm_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        chmm_args, = parser.parse_args_into_dataclasses()

    if chmm_args.log_dir is None:
        chmm_args.log_dir = os.path.join('logs', f'{_current_file_name}.{_time}.log')

    image_cx(args=chmm_args)
