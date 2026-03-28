#!/bin/env python

import argparse
import ast
import re
import shutil
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from tomlkit import TOMLDocument, parse
from tomlkit.items import Table

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
	EDM = "EDM_Loss"
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
	basket: dict[str, str] = dict()

	add_basename(basket, config)
	add_model(basket, config)
	add_optimizer(basket, config)
	add_wavelet(basket, config)
	add_scheduler(basket, config)
	add_ulr(basket, config)
	add_tlr(basket, config)
	add_batch(basket, config)
	add_gradient_accumulation_steps(basket, config)
	add_epoch(basket, config)
	add_step(basket, config)
	add_network(basket, config)
	add_resolution(basket, config)
	add_snr(basket, config)
	add_debias(basket, config)
	add_ipng(basket, config)
	add_edm(basket, config)
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


def li_str_to_dict(li: list[str] | None) -> dict[str, str]:
	if li is None:
		return {}

	ret: dict[str, str] = dict()
	for i in li:
		k, v = i.split("=")
		ret[k] = v

	return ret


def get_basename(config: TOMLDocument) -> str:
	save_group: Table = cast(Table, config.value[GROUPS.SAVE.value])
	return str(save_group["output_name"])


def add_basename(basket: dict[str, str], config: TOMLDocument):
	# basename
	save_group: dict[str, Any] = config.get(GROUPS.SAVE.value)
	basename: str = save_group.get("output_name", "")
	basket[basename] = ""


def add_model(basket: dict[str, str], config: TOMLDocument):
	# model
	basics: dict = config.get(GROUPS.BASICS.value)
	model_name = Path(basics.get("pretrained_model_name_or_path")).stem
	if "noob" in model_name.lower():
		if "vpred" in model_name.lower():
			basket["m"] = "noobv"
		else:
			basket["m"] = "noob"
	if "anima" in model_name.lower():
		basket["m"] = "anima"


def parse_optimizer(optimizer_name: str, optimizer_args: dict) -> str:
	"""Shorthand the optimizer name and arguments.

	Optimizer arguments that are the defaults should not be added to the return
	string.(maybe?)
	"""
	optimizer_name = optimizer_name.lower()
	ret = ""

	if "came" in optimizer_name:
		ret = "CAME"
		optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
		update = optimizer_args.get("update_strategy")
		if update:
			ret += "-uC"

	elif "prodigy" in optimizer_name:
		ret = "Prodigy"

		optimizer_args = li_str_to_dict(optimizer.get("optimizer_args"))
		d_coef = optimizer_args.get("d_coef") or "1"
		ret += f"-{d_coef}"

	elif "fmarscrop" in optimizer_name:
		version = ""
		machina = ""
		mver = re.search(r"v\d", optimizer_name)
		mmach = re.search(r"exmachina", optimizer_name)

		args = ""

		if mver:
			version = f"{mver.group(0)}"

		if mmach:
			machina = "xm"

			update_strategy = optimizer_args.get("update_strategy")
			args += "" if update_strategy in [None, "cautious"] else update_strategy

			if "exmachina" in optimizer_name:
				eps_floor = optimizer_args.get("eps_floor")  # dynamic eps
				args += "" if eps_floor is None else f"epf{eps_floor}"

				moment_centralization = optimizer_args.get(
					"moment_centralization"
				)  # def 0.0
				args += (
					""
					if not moment_centralization
					else f"mc{float(moment_centralization):g}"
				)

		ret = "fmc" + version + machina + (f"-{args}" if args else "")

	elif "compass" in optimizer_name:
		plus = ""
		if re.search(r"Plus", optimizer_name):
			plus = "p"

		ret = "comp" + plus

	elif "schedulefree" in optimizer_name:
		sf_name = ""
		if "radam" in optimizer_name:
			sf_name = "radam"

		ret = "SF" + sf_name

	elif "talon" in optimizer_name:
		payload: str = "TALON"

		signscale_power = optimizer_args.get("signscale_power")
		if signscale_power:
			payload += f"_sp{shorthand_float(signscale_power)}"

		ret = payload

	elif "fftdescent" in optimizer_name:
		payload: str = "fftd"

		if optimizer_args:
			payload += "-"

		beta = optimizer_args.get("beta")
		if beta and beta != 0.95:
			payload += "b" + shorthand_float(beta)

		weight_decay = optimizer_args.get("weight_decay")
		if weight_decay:
			payload += "wd" + shorthand_percent(float(weight_decay))

		if optimizer_args.get("spectral_clip"):
			payload += "NSC"

		if optimizer_args.get("spectral_adaptive"):
			payload += "SA"

		lowpass_grad = optimizer_args.get("lowpass_grad")
		if lowpass_grad and lowpass_grad != 1.0:
			payload += f"lpg{shorthand_float(lowpass_grad)}"

		sign_momentum = optimizer_args.get("sign_momentum")
		if sign_momentum and sign_momentum != 0.9:
			payload += "sm" + shorthand_float(sign_momentum)

		ret = payload

	elif "adamw8bit" in optimizer_name:
		payload = ""
		payload += "adamw8b"

		ret = payload

	elif "adamw" in optimizer_name:
		payload = ""
		payload += "adamw"

		if optimizer_args:
			wd = optimizer_args.get("weight_decay")
			print(wd)
			if wd and wd != 0.01 or wd == 0.0:
				payload += f"_wd{shorthand_percent(float(wd))}"

			b = optimizer_args.get("betas")
			print(b)
			if b and b != (0.9, 0.999):
				payload += f"_b{shorthand_percent(float(b[0]))}b{shorthand_percent(float(b[1]))}"

		print(payload)
		ret += f'_{payload}'

	elif "ocgopt" in optimizer_name:
		payload: str = "ocgopt"

		if optimizer_args:
			payload += "-"

		cntr = optimizer_args.get("centralization")
		if cntr:
			payload += "c" + shorthand_percent(float(cntr))

		c_min = optimizer_args.get("cautious_min")
		if c_min:
			payload += "cm" + shorthand_percent(float(c_min))

		lpg = optimizer_args.get("lowpass_grad")
		if lpg:
			payload += "lpg" + shorthand_percent(float(lpg))

		adpt_min = optimizer_args.get("adaptive_min")
		if adpt_min:
			payload += "amn" + shorthand_percent(float(adpt_min))

		adpt_max = optimizer_args.get("adaptive_max")
		if adpt_max:
			payload += "amx" + shorthand_percent(float(adpt_max))

		if optimizer_args.get("input_norm"):
			payload += "inorm"

		ret = payload

	return ret


def add_optimizer(basket: dict[str, str], config: TOMLDocument):
	# optimizer
	optimizer = config.get(GROUPS.OPTIM.value)
	optimizer_name: str = optimizer.get("optimizer_type").lower()
	optimizer_args: dict = li_str_to_dict(optimizer.get("optimizer_args"))

	basket["o"] = parse_optimizer(optimizer_name, optimizer_args)

	weight_decay = optimizer.get("weight_decay")
	if weight_decay:
		basket["wd"] = str(shorthand_float(weight_decay))

	loss_type: str | None = optimizer.get("loss_type")
	if loss_type and loss_type != "l2":
		basket["l"] = shorthand_loss_type(loss_type)


def shorthand_float(f: float) -> str:
	"""Converts a float to its smallest representation."""
	if int(f) == f:  # simple int, 3.0 -> 3
		return str(int(f))

	if f < 1 and f > -1:
		return str(f).replace(".", "")  # 0.23 -> 023

	return str(f).replace(".", "_")  # 32.32 -> 32_32


def shorthand_percent(f: float) -> str:
	"""Convert a float percentage to a string with no decimal or leading zeros."""
	return str(int(f * 100))


def shorthand_loss_type(lt: str) -> str:
	if lt == "smooth_l2_log":
		return "sl2l"
	if lt == "huber":
		return "h"


def add_wavelet(basket: dict[str, str], config: TOMLDocument):
	optim_conf = config.get(GROUPS.OPTIM.value)
	payload: list[str] = []

	if optim_conf.get("wavelet_loss"):
		payload.append("")
		loss_type: str | None = optim_conf.get("loss_type")

		# if wloss not defined, uses loss. if loss not defined, defaults to l2
		# when wloss is excluded, it's assumed the same as loss
		wloss_type = optim_conf.get("wavelet_loss_type", loss_type) or ""
		wlt = shorthand_loss_type(wloss_type)
		if wlt is None:
			payload[-1] += loss_type if loss_type else "l2"
		elif wloss_type != loss_type:
			payload[-1] += wlt

		wavelet = optim_conf.get("wavelet_loss_wavelet")
		if not wavelet or wavelet != "sym7":
			payload[-1] += wavelet

		loss_level = optim_conf.get("wavelet_loss_level")
		if loss_level and loss_level != 2:
			payload[-1] += "v" + str(loss_level)

		# payload.append("")
		wtrans = optim_conf.get("wavelet_loss_transform")
		if wtrans != "dwt":
			payload[-1] += "f" + wtrans

		walpha = optim_conf.get("wavelet_loss_alpha")
		if walpha and float(walpha) != 0.98:
			payload[-1] += "a" + shorthand_percent(walpha)

		payload.append("")
		l_thresh = optim_conf.get("wavelet_loss_ll_level_threshold")
		if l_thresh:
			payload[-1] += "t" + str(l_thresh).replace("-", "n")

		payload.append("")
		band_raw: str = optim_conf.get("wavelet_loss_band_weights", "")
		if band_raw:
			band: dict[str, float] = ast.literal_eval(band_raw.format())
			payload[-1] += "".join(
				f"{k}{shorthand_percent(v)}" for k, v in band.items()
			)

		quant_raw: str = optim_conf.get("wavelet_loss_quaternion_component_weights")
		if wtrans == "qwt" and quant_raw:
			quant: dict[str, float] = ast.literal_eval(quant_raw.format())
			payload[-1] += "_" + "".join(
				f"{k}{shorthand_percent(v)}" for k, v in quant.items()
			)

		if any(payload):
			basket["w"] = "_".join(p for p in payload if p)


def add_scheduler(basket: dict[str, str], config: TOMLDocument):
	sch_conf = config.get(GROUPS.LR.value)
	sch_type: str | None = sch_conf.get("lr_scheduler_type")

	if sch_type:
		sch_type = sch_type.lower()
		# TODO: all of torch built-in schedulers
		sch_args = li_str_to_dict(sch_conf.get("lr_scheduler_args"))
		if "rex" in sch_type:
			# TODO: figure out what to grab for cycles
			basket["s"] = "rex"

			sch_suffix = ""
			if sch_args:
				d = sch_args.get("d")
				sch_suffix += f"d{float(d) * 10:g}" if d and d != "0.9" else ""

				gamma = sch_args.get("gamma")
				sch_suffix += (
					f"g{float(gamma) * 10:g}" if gamma and gamma != "0.9" else ""
				)

				# cycles = sch_conf.get("lr_scheduler_num_cycles")
				# sch_suffix += f"c{cycles}" if cycles != 1 else ""

				first_cycle_raw = sch_args.get("first_cycle_max_steps", None)
				first_cycle = ast.literal_eval(first_cycle_raw)

				if (
					first_cycle
					and not first_cycle == 1
					and first_cycle != get_max_train_steps(config)
				):
					if isinstance(first_cycle, float):
						sch_suffix += (
							f"fc{shorthand_percent(float(first_cycle), keep_zero=True)}"
						)
					elif isinstance(first_cycle, int):
						sch_suffix += f"fc{first_cycle}"

				warmup = sch_args.get("warmup_steps")
				sch_suffix += (
					f"w{float(warmup):g}" if warmup and float(warmup) != 0 else ""
				)

				min_lr = sch_args.get("min_lr")
				if min_lr:
					min_lr = float(min_lr)
					sch_suffix += (
						f"m{notation_normalize(min_lr)}" if min_lr != 0.000001 else ""
					)

				if sch_suffix:
					basket[sch_suffix] = ""

		elif "cos" in sch_type:
			basket["s"] = "cos"

			sch_suffix = ""
			if sch_args:
				d = sch_args.get("d")
				gamma = sch_args.get("gamma")
				cycles = sch_conf.get("lr_scheduler_num_cycles")
				warmup = sch_args.get("warmup_steps")
				min_lr = sch_args.get("min_lr")
				min_lr = float(min_lr) if float(min_lr) else None

				if min_lr and min_lr != 1e-6:
					sch_suffix += f"m{notation_normalize(min_lr)}"

				if warmup and float(warmup) != 0:
					sch_suffix += f"w{float(warmup):g}"

				if d and d != "0.9":
					sch_suffix += f"d{shorthand_percent(d)}"

				if gamma and gamma != "0.9":
					sch_suffix += f"g{float(gamma) * 10:g}"

				if cycles:
					sch_suffix += f"c{cycles}" if cycles != 1 else ""

				if sch_suffix:
					basket[sch_suffix] = ""

	else:
		sch = sch_conf.get("lr_scheduler")
		if "linear" in sch:
			basket["s"] = "lin"
		if "constant" in sch:
			basket["s"] = "const"
		if "cosine" in sch:
			basket["s"] = "cos"

		sch_suffix = ""
		warmup = sch_conf.get("lr_warmup_steps")
		if warmup:
			if isinstance(warmup, float):
				sch_suffix += f"w{shorthand_float(warmup)}"
			elif isinstance(warmup, int):
				sch_suffix += f"w{warmup}"

		if sch_suffix:
			basket[sch_suffix] = ""


def add_ulr(basket: dict[str, str], config: TOMLDocument):
	"""Learning rate will always be present."""
	optimizer = config.get(GROUPS.OPTIM.value)
	ulr = optimizer.get("unet_lr")
	basket["u"] = notation_normalize(ulr)


def add_tlr(basket: dict[str, str], config: TOMLDocument):
	"""Learning rate will always be present."""
	optimizer = config.get(GROUPS.OPTIM.value)
	network_setup = config.get("Network_setup")
	if not network_setup.get("network_train_unet_only"):
		tlr = optimizer.get("text_encoder_lr")
		basket["t"] = notation_normalize(tlr)


def add_batch(basket: dict[str, str], config: TOMLDocument):
	# batch
	optimizer = config.get(GROUPS.OPTIM.value)
	batch = optimizer.get("train_batch_size", 1)
	if batch > 1:
		basket["b"] = str(batch)


def add_gradient_accumulation_steps(basket: dict[str, str], config: TOMLDocument):
	# gradient accumulation
	optimizer = config.get(GROUPS.OPTIM.value)
	steps = optimizer.get("gradient_accumulation_steps", 1)
	if int(steps) > 1:
		basket["g"] = str(steps)


def add_epoch(basket: dict[str, str], config: TOMLDocument):
	# epoch
	basics = config.get(GROUPS.BASICS.value)
	epoch = basics.get("max_train_epochs")
	if epoch:
		basket["e"] = str(epoch)


def get_max_train_steps(config: TOMLDocument):
	basics = config.get(GROUPS.BASICS.value)
	steps = basics.get("max_train_steps", None)
	return int(steps)


def add_step(basket: dict[str, str], config: TOMLDocument):
	steps = get_max_train_steps(config)
	if steps and "e" not in basket:
		basket["st"] = str(steps)


def add_network(basket: dict[str, str], config: TOMLDocument):
	# network
	lyco = config.get(GROUPS.LYCO.value)
	module = lyco.get("network_module")
	network_args = lyco.get("network_args")
	full_matrix = False

	network_args = li_str_to_dict(network_args)

	if "networks.lora" == module:
		basket["a"] = "lora"

	elif "lycoris.kohya" == module:
		algo = network_args.get("algo")
		if algo == "lokr":
			if network_args.get("dora_wd"):
				algo = "d" + algo[1:]

			if network_args.get("wd_on_output"):
				algo += "r"

			basket["a"] = algo

			algo_suffix = ""
			factor = network_args.get("factor")
			full_matrix = network_args.get("full_matrix")

			algo_suffix += f"f{factor}" if factor else ""
			algo_suffix += "fm" if full_matrix else ""

			basket["".join(algo_suffix)] = ""

		elif algo == "locon":
			if network_args.get("dora_wd"):
				algo = "docon"
				if network_args.get("wd_on_output"):
					algo = "ddocon"
			basket["a"] = algo

		elif algo == "glora":
			basket["a"] = algo

	preset = network_args.get("preset")  # default is full, do not add
	if preset == "full-lin":
		preset = "fl"
	elif preset == "attn-mlp":
		preset = "am"
	elif preset == "attn-only":
		preset = "ao"
	elif preset == "unet-transformer-only":
		preset = "uto"
	elif preset == "unet-convblock-only":
		preset = "wco"

	if preset:
		basket["p"] = preset

	# dim/alpha
	#
	# dim/alpha are set by set by lokr factor, settings are irrelevant here?
	# if algo not in ["lokr", "dokr"]:
	network = config.get(GROUPS.NET.value)
	rank = ""

	rank += f"d{network.get('network_dim', 4)}" if not full_matrix else ""
	rank += f"a{network.get('network_alpha', 1)}"

	if network_args:
		rank += f"cd{network_args.get('conv_dim', 4)}" if not full_matrix else ""
		rank += f"ca{network_args.get('conv_alpha', 1)}"

		rs_lora = network_args.get("rs_lora")
		if rs_lora:
			rank += "rs"

		basket[rank] = ""

	# other LoRA adjustments

	# lora plus
	lorap = network_args.get("loraplus_lr_ratio") if network_args else None
	if lorap:
		basket["lp"] = shorthand_float(float(lorap))

	# using scalar
	scalar = network_args.get("use_scalar") if network_args else None
	if scalar:
		basket["sclr"] = ""


def add_snr(basket: dict[str, str], config: TOMLDocument):
	# other improvements, else
	improvements = config.get(GROUPS.IMPROV.value)
	snr = improvements.get("min_snr_gamma")
	if snr:
		basket["snr"] = str(snr)


def add_ipng(basket: dict[str, str], config: TOMLDocument):
	improvements = config.get(GROUPS.IMPROV.value)
	ipng = improvements.get("ip_noise_gamma")
	if ipng:
		basket["ip"] = f"{ipng * 10:.1g}"


def add_edm(basket: dict[str, str], config: TOMLDocument):
	edm = config.get(GROUPS.EDM.value)
	if not edm:
		return

	enabled = edm.get("edm2_loss_weighting")
	if enabled:
		payload = ""

		ch = edm.get("edm2_loss_weighting_num_channels")
		if ch and int(ch) != 128:
			payload += f"c{ch}"

		opt = edm.get("edm2_loss_weighting_optimizer")
		if opt and opt != "torch.optim.AdamW":
			payload += "_oocgopt"

		lr = edm.get("edm2_loss_weighting_optimizer_lr")
		if lr and float(lr) != 0.02:
			payload += f"_l{notation_normalize(float(lr))}"

		important_enabled = edm.get("edm2_loss_weighting_importance_weighting")
		if important_enabled:
			payload += "_imp"

			imax = edm.get("edm2_loss_weighting_importance_weighting_max")
			if imax and float(imax) != 10.0:
				payload += f"m{shorthand_float(imax)}"

			snr = edm.get("edm2_loss_weighting_importance_weighting_min_snr_gamma")
			if snr and float(snr) != 1.0:
				payload += f"s{shorthand_float(snr)}"

		print(payload)

		optimizer_name = edm.get("edm2_loss_weighting_optimizer", "AdamW")
		optimizer_args = edm.get("edm2_loss_weighting_optimizer_args", "{}")
		optimizer_args = re.match(r"{(.*)}", optimizer_args).group(1)
		if optimizer_args:
			optimizer_args = ast.literal_eval(optimizer_args)

		res: str = parse_optimizer(optimizer_name, optimizer_args)

		res = res.replace('adamw_','')
		res = res.replace('_adamw','')
		res = res.replace('adamw','')

		payload += f"{res}" if res else ""

		basket["edm"] = payload


def add_debias(basket: dict[str, str], config: TOMLDocument):
	improvements = config.get(GROUPS.IMPROV.value)
	debiased = improvements.get("debiased_estimation_loss")
	if debiased:
		basket["db"] = ""


def add_resolution(basket: dict[str, str], config: TOMLDocument):
	# dataset and resolution
	basics = config.get(GROUPS.BASICS.value)
	training_resolution = basics.get("resolution")
	if training_resolution != "1024":
		basket["r"] = training_resolution


def add_dataset(basket: dict[str, str], dataset: TOMLDocument):
	dataset_general = dataset.get("general")
	datasets = dataset.get("datasets")
	dataset_names: list[str] = []
	for dataset in datasets:
		keep = dataset.get("keep_tokens")
		keep = keep if keep != 0 else keep
		subsets = dataset.get("subsets")
		subset_names: list[str] = []
		for subset in subsets:
			payload = ""
			dataset_name = Path(subset.get("image_dir")).stem
			payload += dataset_name
			repeats = subset.get("num_repeats")
			payload += f"r{repeats}" if repeats and repeats != 1 else ""

			crop = subset.get("random_crop")
			if crop:
				payload += "c"
				random_crop_padding_percent = subset.get("random_crop_padding_percent")
				payload += shorthand_float(random_crop_padding_percent)

			subset_names.append(payload)

		dataset_names.append((f"k{keep}" if keep else "") + "_".join(subset_names))

	datasets_subsets_joined = "-".join(dataset_names)

	res = ""
	if dataset_general:
		dataset_resolution = dataset_general.get("resolution")
		if dataset_resolution != 1024:
			res = f"R{dataset_resolution}"

	basket[f"{f'-{res}' if res else ''}{datasets_subsets_joined}"] = ""


def is_find_lr(config: TOMLDocument):
	validation = config.get("Validation")
	if validation:
		return validation.get("is_find_lr")

	return False


def get_resume(config: TOMLDocument):
	network = config.get("Network_setup")
	if network:
		return network.get("resume")


def get_network_weights(config: TOMLDocument):
	network = config.get("Network_setup")
	if network:
		return network.get("network_weights")


def diff_basket_names(b1: str, b2: str):
	"""Return bakset elements of b1 not in b2"""
	b1_li = b1.split("-")
	b2_li = b2.split("-")

	for e2 in b2_li:
		if e2 in b1_li:
			b1_li.remove(e2)

	return "-".join(b1_li)


def natural_stem(p: Path) -> list:
	"""Chunk stem path by word-number for sorting.

	Python sorts by lexicograph, which is no good for humans. The general idea
	for this program is to watch a directory, and will choose the first time
	sorted naturally.

	This /works/ by chunking the stem into by string and number, put into a
	list. Numbers are forced to be ints so it gets sorted by it's numerical
	value rather than lexico, and string sorting works as expected.
	"""
	s = p.stem
	return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]


def exists_handler(p: Path) -> Path:
	"""Return a renamed path if exists, otherwise return unchanged"""
	if not p.exists():
		return p

	name = p.stem
	ext = p.suffix

	number = 1
	if "_" in p.stem:
		try:
			name_split = name.split("_")
			number = int(name_split[-1])
			number += 1
			name = "_".join(name_split[:-1])
		except ValueError:
			pass

	return p.parent / f"{name}_{number}{ext}"


def main():
	jobs_path: Path = args.jobs_path
	failed_jobs: set = set()

	while True:
		jobs = sorted(list(jobs_path.iterdir()), key=natural_stem)
		job = None

		while jobs:
			_job = jobs.pop(0)
			if _job in failed_jobs:
				continue

			job = _job
			break

		if job is None:
			break

		# must match pattern
		config_file = job / "config.toml"
		dataset_file = job / "dataset.toml"

		try:
			with open(config_file, "r") as fp:
				config_file_content = fp.read()
				config = parse(config_file_content)

			with open(dataset_file, "r") as fp:
				dataset_file_content = fp.read()
				dataset = parse(dataset_file_content)
		except (FileNotFoundError, NotADirectoryError):
			failed_jobs.add(job)
			continue

		basket = prepare_basket(config, dataset)
		basename = get_basename(config)
		name = "-".join(f"{k}{v}" for k, v in basket.items())
		name = name.replace(".", "_")  # convert decimals to _

		if is_find_lr(config):  # for temp trial runs for finding lr
			name = "find_lr-" + name

		# save to proper dir if resume, else normal
		#
		# a training set B is nested within a training set A, it should be
		# assumed that B is a continued training from A
		#
		# slight problem is that there my be a clash in naming scheme if
		# there are other models ran with the same parameters, but from
		# scratch, i guess we can just append name of the OG it's resuming
		# from. it'll look ugly but it'll work and guarantee a unique name...
		#
		# we append the difference in continuation parameters, much more elegant
		continue_from = get_network_weights(config) or get_resume(config)
		if continue_from:
			continue_from = Path(continue_from)
			continue_from_name = continue_from.stem

			# shorthand for use the final trained model
			# if continue_from_name == "model":
			# 	continue_from_name = continue_from.parent.stem

			diff_name = ""

			from_steps_match = re.search(r"-step(\d+)", continue_from_name)
			if from_steps_match:
				from_steps = int(from_steps_match.groups()[-1])
				diff_name += f"from{from_steps}"
			else:
				from_steps = 0

			diff_name += diff_basket_names(name, continue_from_name) or ""

			if get_network_weights(config):
				name = f"{continue_from_name}-NETWEIGHTS{('-' + diff_name) if diff_name else '-same'}"
				output_dir = Path(continue_from).parent / f"NETWEIGHTS-{diff_name}"
			elif get_resume(config):
				name = f"{continue_from_name}-RESUME{('-' + diff_name) if diff_name else ''}"
				output_dir = (
					Path(continue_from).parent
					/ f"RESUME{('-' + diff_name) if diff_name else ''}"
				)

			# print(f"{str(output_dir)=}")

			# if not diff_name:
			# 	name = f"{continue_from_name}"
			# 	output_dir = Path(continue_from).parent
		else:
			output_dir = Path(
				config[GROUPS.SAVE.value]["output_dir"].format(
					basename=basename, name=name
				)
			)

		output_dir = Path(
			config[GROUPS.SAVE.value]["output_dir"].format(basename=basename, name=name)
		)

		output_file = output_dir / f"{name}.safetensors"

		# make sure output file does not already exist
		if output_file.exists():
			failed_jobs.add(job)
			print(
				f"the following output model already exists, skipping...\n{output_file}"
			)
			continue

		# confirm name
		print(
			f"training under the following name. make sure theses params are correct for this session:\n{name}\n"
		)

		# modify in-memory toml content from earlier read and do global sub
		new_config = config_file_content.format(
			name=name,
			basename=basename,
			datetime=datetime.now().strftime("%Y%m%d_%H%M%S"),
		)

		# if we're continuing from a previous training, overwrite output_dir
		# with our nested training session
		new_config = re.sub(
			"output_dir = .*", f'output_dir = "{output_dir}"', new_config
		)

		# change output name, which was basename, to name
		new_config = re.sub(
			re.escape(f'output_name = "{basename}"'),
			f'output_name = "{name}"',
			new_config,
		)

		new_dataset = dataset_file_content.format(basename=basename)

		# start training
		dry = args.dry
		if dry:
			failed_jobs.add(job)
			continue

		# write to runtime store
		runtime = Path("./backend/runtime_store/")
		runtime.mkdir(exist_ok=True)
		with open(runtime / "config.toml", "w") as fp:
			fp.write(new_config)

		with open(runtime / "dataset.toml", "w") as fp:
			fp.write(new_dataset)

		command = [
			"python",
			"./backend/sd_scripts/anima_train_network.py",
			"--dataset_config",
			"./backend/runtime_store/dataset.toml",
			"--config_file",
			"./backend/runtime_store/config.toml",
		]
		try:
			proc = subprocess.run(command, check=True)
			if proc.returncode == 0:
				archive_dir = job.parent.parent / "archive" / basename
				archive_dir.mkdir(parents=True, exist_ok=True)
				dst = archive_dir / name
				if dst.exists():
					shutil.move(dst, exists_handler(dst / "old"))
				shutil.move(job, archive_dir / name)  # also renames it appropriately
		except Exception as e:
			failed_jobs.add(job)
			print(e)


if __name__ == "__main__":
	main()
