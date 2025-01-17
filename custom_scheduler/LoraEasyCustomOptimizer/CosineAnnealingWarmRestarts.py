# ripped from derrian-distro https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend/tree/92e696b0bd47bab5742dd695f5848da7a5dee8c2/custom_scheduler
# updated for torch 2.7 nightly
from functools import wraps
import math
import weakref
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


# optimizer, cycle multiplier, and gamma are constant so they should be passed in no matter what
# the rest are either used if last_epoch = -1 and are not already in the param groups or not used if otherwise
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        cycle_multiplier: float = 1,
        first_cycle_max_steps: int = 1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.cycle_multiplier = cycle_multiplier
        self.gamma = gamma  # debating calling this decay_rate or something
        self.last_epoch = last_epoch

        # new run
        if last_epoch == -1:
            if warmup_steps >= first_cycle_max_steps:
                raise ValueError(
                    f"[-] warmup_steps must be smaller than first_cycle_max_steps. "
                    f"{warmup_steps} < {first_cycle_max_steps}"
                )
            self.setup_optimizer(warmup_steps, first_cycle_max_steps, min_lr)
        self.validate_optimizer()

        # def with_counter(method):
        #     if getattr(method, "_with_counter", False):
        #         return method
        #     instance_ref = weakref.ref(method.__self__)
        #     func = method.__func__
        #     cls = instance_ref().__class__
        #     del method
        #
        #     @wraps(func)
        #     def wrapper(*args, **kwargs):
        #         instance = instance_ref()
        #         instance._step_count += 1
        #         wrapped = func.__get__(instance, cls)
        #         return wrapped(*args, **kwargs)
        #
        #     wrapper._with_counter = True
        #     return wrapper
        #
        # self.optimizer.step = with_counter(self.optimizer.step)
        #
        # self._initial_step()

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                opt_ref = weakref.ref(self.optimizer)
                func = step_fn.__func__

                @wraps(func)
                def wrapper(*args, **kwargs):
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        patch_track_step_called(self.optimizer)
        self._initial_step()

    def setup_optimizer(
        self,
        warmup_steps: int,
        first_cycle_max_steps: int,
        min_lr: float,
    ) -> Optimizer:
        for group in self.optimizer.param_groups:
            if "warmup_steps" not in group:
                group.setdefault("warmup_steps", warmup_steps)
            if "current_cycle_max_steps" not in group:
                group.setdefault("current_cycle_max_steps", first_cycle_max_steps)
            if "min_lr" not in group:
                group.setdefault("min_lr", min_lr)
            group.setdefault("current_cycle", 0)
            group.setdefault("current_cycle_step", -1)
            group.setdefault("initial_lr", group["lr"])
            group.setdefault("current_max_lr", group["lr"])

    def validate_optimizer(self):
        for i, group in enumerate(self.optimizer.param_groups):
            for key in {
                "warmup_steps",
                "current_cycle_max_steps",
                "min_lr",
                "current_cycle",
                "current_cycle_step",
                "initial_lr",
                "current_max_lr",
            }:
                if key not in group:
                    raise KeyError(
                        f"param '{key}' is not specified in param_groups[{i}] when resuming an optimizer"
                    )
            if group["warmup_steps"] >= group["current_cycle_max_steps"]:
                raise ValueError(
                    f"[-] warmup_steps must be smaller than first_cycle_max_steps. "
                    f"{group['warmup_steps']} < {group['current_cycle_max_steps']}"
                )

    def _calc_first_step(self, group: list[float | int]):
        while group["current_cycle_step"] >= group["current_cycle_max_steps"]:
            group = self._update_cycle(group)
        return group

    def _update_step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            if group["current_cycle_step"] == -1:
                group = self._calc_first_step(group)
                self.optimizer.param_groups[i] = group
            group["current_cycle_step"] += 1
            group = self._update_cycle(group)
            self.optimizer.param_groups[i] = group

    def _update_cycle(self, group: list[float | int]):
        if group["current_cycle_step"] < group["current_cycle_max_steps"]:
            return group
        group["current_cycle_step"] -= group["current_cycle_max_steps"]
        group["current_cycle"] += 1
        group["current_cycle_max_steps"] = (
            round(
                (group["current_cycle_max_steps"] - group["warmup_steps"])
                * self.cycle_multiplier
            )
            + group["warmup_steps"]
        )
        group["current_max_lr"] = group["initial_lr"] * (
            self.gamma ** group["current_cycle"]
        )
        return group

    def get_lr(self) -> float:
        self._update_step()
        lrs = []
        for group in self.optimizer.param_groups:
            if group["current_max_lr"] <= group["min_lr"]:
                lrs.append(group["min_lr"])
                continue
            lr_range = group["current_max_lr"] - group["min_lr"]
            if group["current_cycle_step"] < group["warmup_steps"]:
                lrs.append(
                    lr_range * group["current_cycle_step"] / group["warmup_steps"]
                    + group["min_lr"]
                )
                continue
            normalized_step = group["current_cycle_step"] - group["warmup_steps"]
            normalized_max_steps = (
                group["current_cycle_max_steps"] - group["warmup_steps"]
            )
            lrs.append(
                lr_range
                * (1 + math.cos(math.pi * normalized_step / normalized_max_steps))
                / 2.0
                + group["min_lr"]
            )
        return lrs
