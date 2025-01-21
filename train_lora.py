#!/bin/env python

import argparse
from tomlkit import parse, TOMLDocument
from pathlib import Path
import subprocess
import re
import sys
import shutil
from datetime import datetime
from enum import Enum

parser = argparse.ArgumentParser()

parser.add_argument("jobs_path", type=Path, help="location of config and dataset toml")
parser.add_argument(
    "-d",
    "--dry",
    action=argparse.BooleanOptionalAction,
    help="only show resulting name (training parameters)",
)

args = parser.parse_args()


class GROUPS(Enum):
    BASICS = "Basics"
    SAVE = "Save"
    SDv2 = "SDv2"
    NET = "Network_setup"
    LYCO = "LyCORIS"
    OPTIM = "Optimizer"
    LR = "Lr_scheduler"
    PRECISION = "Training_precision"
    IMPROV = "Further_improvement"
    ARB = "ARB"
    CAP = "Captions"
    ATTN = "Attention"
    AUG = "Data_augmentation"
    CACHE = "Cache_latents"
    SAMP = "Sampling_during_training"
    LOG = "Logging"
    REG = "Regularization"
    HUGGING = "Huggingface"
    DEBUG = "Debugging"
    DEPRECATED = "Deprecated"
    OTHER = "Others"


def prepare_basket(config: TOMLDocument, dataset: TOMLDocument):
    basket: list[list[str:str]] = []

    add_basename(basket, config)
    add_model(basket, config)
    add_optimizer(basket, config)
    add_ulr(basket, config)
    add_tlr(basket, config)
    add_batch(basket, config)
    add_epoch(basket, config)
    add_network(basket, config)
    add_resolution(basket, config)
    add_snr(basket, config)
    add_debias(basket, config)
    add_ipng(basket, config)
    add_dataset(basket, dataset)

    return basket


def notation_normalize(n: float) -> str:
    """Convert float to scientific notation fixed to an exponent.

    Example:
    1.6e-4 -> 16e-5
    """
    if n == 1:
        return "1"

    c, e = f"{n:e}".split("e")
    c = c.rstrip("0").rstrip(".")
    e = e[1:].lstrip("0")  # slice to remove negative sign

    return f"{c}e{e}"


def li_str_to_dict(li: list[str]) -> dict:
    ret = dict()
    for i in li:
        k, v = i.split("=")
        ret[k] = v

    return ret


def get_basename(config: TOMLDocument) -> str:
    return config[GROUPS.SAVE.value]["output_name"]


def add_basename(basket: dict, config: TOMLDocument):
    # basename
    print(GROUPS.SAVE.value)
    basename = config[GROUPS.SAVE.value]["output_name"]
    basket.append(["", basename])


def add_model(basket: dict, config: TOMLDocument):
    # model
    basics = config.get(GROUPS.BASICS.value)
    model_name = Path(basics.get("pretrained_model_name_or_path")).stem
    if "noob" in model_name.lower():
        if "vpred" in model_name.lower():
            basket.append(["m", "noobv"])
        else:
            basket.append(["m", "noob"])


def add_optimizer(basket: dict, config: TOMLDocument):
    # optimizer
    optimizer = config.get(GROUPS.OPTIM.value)
    optimizer_name = optimizer.get("optimizer_type")

    if "came" in optimizer_name.lower():
        basket.append(["o", "CAME"])

    elif "prodigy" in optimizer_name.lower():
        basket.append(["o", "Prodigy"])

        optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
        d_coef = optimizer_args.get("d_coef") or "1"
        basket.append(["d", d_coef])


def add_ulr(basket: dict, config: TOMLDocument):
    # unet lr
    optimizer = config.get(GROUPS.OPTIM.value)
    ulr = optimizer.get("unet_lr")
    basket.append(["u", notation_normalize(ulr)])


def add_tlr(basket: dict, config: TOMLDocument):
    # te lr
    optimizer = config.get(GROUPS.OPTIM.value)
    network_setup = config.get("Network_setup")
    if not network_setup.get("network_train_unet_only"):
        tlr = optimizer.get("text_encoder_lr")
        basket.append(["t", notation_normalize(tlr)])


def add_batch(basket: dict, config: TOMLDocument):
    # batch
    optimizer = config.get(GROUPS.OPTIM.value)
    batch = optimizer.get("train_batch_size")
    basket.append(["b", str(batch)])


def add_epoch(basket: dict, config: TOMLDocument):
    # epoch
    basics = config.get(GROUPS.BASICS.value)
    epoch = basics.get("max_train_epochs")
    basket.append(["e", str(epoch)])


def add_network(basket: dict, config: TOMLDocument):
    # network
    lyco = config.get(GROUPS.LYCO.value)
    module = lyco.get("network_module")
    network_args = lyco.get("network_args")
    if network_args:
        parsed_network = li_str_to_dict(network_args)

    if "networks.lora" == module:
        basket.append(["a", "lora"])
        # I have been told there is little-to-no difference between training
        # vs inference lbw. So we will not care about this anymore
        # if parsed_network.get("down_lr_weight"):
        #     weights = parsed_network["down_lr_weight"].split(",")
        #     basket.append(["wd", weights[0]])

    elif "lycoris.kohya" == module:
        algo = parsed_network.get("algo")
        if algo == "lokr":
            if "wd_on_output" in parsed_network:
                basket.append(["a", "dokr"])
            else:
                basket.append(["a", "lokr"])

            if "factor" in parsed_network:
                factor = parsed_network["factor"]
            else:
                factor = 0
            basket.append(["f", str(factor)])
        elif algo == "locon":
            basket.append(["a", "locon"])

    # dim/alpha
    network = config.get(GROUPS.NET.value)

    dim = int(network.get("network_dim", 4))
    alpha = int(network.get("network_alpha", 1))

    if network_args:
        cdim = int(parsed_network.get("conv_dim", 4))
        calpha = int(parsed_network.get("conv_alpha", 1))

    basket.append(
        [
            "",
            f"d{dim}a{alpha}"
            if not network_args
            else f"d{dim}a{alpha}cd{cdim}ca{calpha}",
        ]
    )


def add_snr(basket: dict, config: TOMLDocument):
    # other improvements, else
    improvements = config.get(GROUPS.IMPROV.value)
    snr = improvements.get("min_snr_gamma")
    if snr:
        basket.append(["snr", str(snr)])


def add_ipng(basket: dict, config: TOMLDocument):
    improvements = config.get(GROUPS.IMPROV.value)
    ipng = improvements.get("ip_noise_gamma")
    if ipng:
        basket.append(["ip", f"{ipng * 10:.1g}"])


def add_debias(basket: dict, config: TOMLDocument):
    improvements = config.get(GROUPS.IMPROV.value)
    debiased = improvements.get("debiased_estimation_loss")
    if debiased:
        basket.append(["db", ""])


def add_resolution(basket: dict, config: TOMLDocument):
    # dataset and resolution
    basics = config.get(GROUPS.BASICS.value)
    training_resolution = basics.get("resolution")
    if training_resolution != "1024":
        basket.append(["r", training_resolution])


def add_dataset(basket: dict, dataset: TOMLDocument):
    dataset_general = dataset.get("general")
    dataset_name = Path(dataset["datasets"][0]["subsets"][0]["image_dir"]).stem

    if dataset_general:
        dataset_resolution = dataset_general.get("resolution")
        if dataset_resolution != 1024:
            ds = dataset_name + f"r{dataset_resolution}"
        else:
            ds = dataset_name

    basket.append(["", ds])


def main():
    jobs: Path = args.jobs_path
    for job in jobs.iterdir():
        config_file = job / "config.toml"
        dataset_file = job / "dataset.toml"

        with open(config_file, "r") as fp:
            config_file_content = fp.read()
            config = parse(config_file_content)

        with open(dataset_file, "r") as fp:
            dataset_file_content = fp.read()
            dataset = parse(dataset_file_content)

        basket = prepare_basket(config, dataset)
        basename = get_basename(config)
        name = "-".join("".join((k, v)) for k, v in basket)
        name = name.replace(".", "_")  # convert decimals to _

        # make sure output file does not already exist
        output_dir = Path(
            config[GROUPS.SAVE.value]["output_dir"].format(basename=basename, name=name)
        )
        output_file = output_dir / f"{name}.safetensors"
        if output_file.exists():
            print(f'Output model "{output_file}" already exists, skipping...')
            sys.exit(1)

        # confirm name
        print(
            f"training under the following name. make sure theses params are correct for this session:\n{name}"
        )

        # modify in-memory toml content from earlier read and do global sub
        new_config = config_file_content.format(
            name=name,
            basename=basename,
            datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        # change output name, which was basename, to name
        new_config = re.sub(
            re.escape(f'output_name = "{basename}"'),
            f'output_name = "{name}"',
            new_config,
        )

        new_dataset = dataset_file_content.format(basename=basename)

        # write to runtime store
        runtime = Path("./runtime_store/")
        runtime.mkdir(exist_ok=True)
        with open(runtime / "config.toml", "w") as fp:
            fp.write(new_config)

        with open(runtime / "dataset.toml", "w") as fp:
            fp.write(new_dataset)

        # start training
        dry = args.dry
        if dry:
            continue

        command = [
            "python",
            "./sdxl_train_network.py",
            "--dataset_config",
            "./runtime_store/dataset.toml",
            "--config_file",
            "./runtime_store/config.toml",
        ]
        try:
            result = subprocess.run(command, check=True)
            if result:
                archive_dir = job.parent.parent / "archive" / basename
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(job, archive_dir / name)  # also renames it appropriately
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
