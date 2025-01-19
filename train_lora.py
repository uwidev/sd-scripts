#!/bin/env python

import argparse
from tomlkit import parse
from pathlib import Path
import subprocess
import re
import sys
import shutil
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("jobs_path", type=Path, help="location of config and dataset toml")
parser.add_argument(
    "-d",
    "--dry",
    action=argparse.BooleanOptionalAction,
    help="only show resulting name (training parameters)",
)

args = parser.parse_args()

jobs: Path = args.jobs_path


# def notation_normalize(n: float, to_exp: int = 5) -> str:
#     """Convert float to scientific notation fixed to an exponent.
#
#     Example:
#     1.6e-4 -> 16e-5
#     """
#     if n == 1:
#         return "1"
#
#     c, e = f"{n:.{to_exp}e}".split("e")
#     diff = to_exp - -int(e)
#     c = float(c) * 10**diff
#     return f"{c:n}e{to_exp}"


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


for job in jobs.iterdir():
    config_file = job / "config.toml"
    dataset_file = job / "dataset.toml"

    with open(config_file, "r") as fp:
        config_file_content = fp.read()
        config = parse(config_file_content)

    with open(dataset_file, "r") as fp:
        dataset_file_content = fp.read()
        dataset = parse(dataset_file_content)

    # gather data for naming
    basket: list[list[str:str]] = []

    # basename
    basename = config["Save"]["output_name"]
    basket.append(["", basename])

    # model
    basics = config.get("Basics")
    model_name = Path(basics.get("pretrained_model_name_or_path")).stem
    if "noob" in model_name.lower():
        if "vpred" in model_name.lower():
            basket.append(["m", "noobv"])
        else:
            basket.append(["m", "noob"])

    # optimizer
    optimizer = config.get("Optimizer")
    optimizer_name = optimizer.get("optimizer_type")

    if "came" in optimizer_name.lower():
        basket.append(["o", "CAME"])

    elif "prodigy" in optimizer_name.lower():
        basket.append(["o", "Prodigy"])

        optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
        d_coef = optimizer_args.get("d_coef") or "1"
        basket.append(["d", d_coef])

    # unet lr
    ulr = optimizer.get("unet_lr")
    basket.append(["u", notation_normalize(ulr)])
    # basket.append(["u", f"{ulr:.1e}"])

    # te lr
    network_setup = config.get("Network_setup")
    if not network_setup.get("network_train_unet_only"):
        tlr = optimizer.get("text_encoder_lr")
        basket.append(["t", notation_normalize(tlr)])
        # basket.append(["t", f"{tlr:.0e}"])

    # batch
    batch = optimizer.get("train_batch_size")
    basket.append(["b", str(batch)])

    # epoch
    epoch = basics.get("max_train_epochs")
    basket.append(["e", str(epoch)])

    # network
    lyco = config.get("LyCORIS")
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
        if parsed_network.get("algo") == "lokr":
            if "wd_on_output" in parsed_network:
                basket.append(["a", "dokr"])
            else:
                basket.append(["a", "lokr"])

            if "factor" in parsed_network:
                factor = parsed_network["factor"]
            else:
                factor = 0
            basket.append(["f", str(factor)])

    # other improvements, else
    improvements = config.get("Further_improvement")
    snr = improvements.get("min_snr_gamma")
    if snr:
        basket.append(["snr", str(snr)])

    ipng = improvements.get("ip_noise_gamma")
    if ipng:
        basket.append(["ip", f"{ipng * 10:.1g}"])

    debiased = improvements.get("debiased_estimation_loss")
    if debiased:
        basket.append(["db", ""])

    # dataset and resolution
    training_resolution = basics.get("resolution")
    if training_resolution != "1024":
        basket.append(["r", training_resolution])

    dataset_general = dataset.get("general")
    dataset_name = Path(dataset["datasets"][0]["subsets"][0]["image_dir"]).stem

    if dataset_general:
        dataset_resolution = dataset_general.get("resolution")
        if dataset_resolution != 1024:
            ds = dataset_name + f"r{dataset_resolution}"
        else:
            ds = dataset_name

    basket.append(["", ds])

    name = "-".join("".join((k, v)) for k, v in basket)

    # make sure output file does not already exist
    output_dir = Path(config["Save"]["output_dir"].format(basename=basename, name=name))
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
        name=name, basename=basename, datetime=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # change output name, which was basename, to name
    new_config = re.sub(
        re.escape(f'output_name = "{basename}"'), f'output_name = "{name}"', new_config
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
