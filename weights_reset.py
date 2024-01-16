from typing import Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightsReset(nn.Module):
    """Weights reset.

    Implements the weights reset regularizer (Plusch et al., 2023). Weights 
    reset periodically resets a portion of the weights in a layer. Intuitively,
    this aims to nudge the model out of local minima.

    Example
    -------
    >>> module = WeightsReset(
    ...     start=0,
    ...     stop=100_000,
    ...     interval=100,
    ...     rate=0.1,
    ...     initializer=nn.init.kaiming_normal,
    ...     module=nn.Linear(256, 256),
    ... )
    """

    def __init__(
        self,
        *,
        start: int,
        stop: int,
        interval: int,
        rate: float,
        initializer: Callable,
        module: nn.Module,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        start : int
            The number of forward steps before weights reset is enabled.
        stop : int
            The number of forward steps before weights reset is disabled.
        interval : int
            The number of forward steps between resets.
        rate : int
            The probability that a weight will be reset.
        initializer : Callable
            A `torch.nn.init` initializer, used to reset weights.
        module : nn.Module
            The target module.
        """

        super().__init__()

        self.start = start
        self.stop = stop
        self.interval = interval
        self.rate = rate
        self.module = module
        self.initializer = initializer
        self.step = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass."""
        
        self.step += 1
        
        # Apply weights reset.

        if self.training \
            and (self.step in range(self.start, self.stop + 1)) \
            and (self.step % self.interval == 0):
            
            with torch.no_grad():
                for parameter in self.module.parameters():

                    mask = (torch.rand_like(parameter) < self.rate).to(parameter.device)
                    parameter.mul_(torch.logical_not(mask))

                    if len(parameter.shape) >= 2: 
                        parameter.add_(self.initializer(parameter) * mask)

        return self.module(*args, **kwargs)


@dataclass(frozen=True)
class WeightsResetV2Configuration:
    """Weights reset v2 configuration."""

    start: int
    stop: int
    interval: int
    rate: float
    initializer: Callable


class WeightsResetV2(nn.Module):
    """Weights reset v2.

    Example
    -------
    >>> configuration = WeightsResetV2Configuration(
    ...     start=0,
    ...     stop=100_000,
    ...     interval=100,
    ...     rate=0.1,
    ...     initializer=nn.init.kaiming_normal,
    ...     module=nn.Linear(256, 256),
    ... )
    >>> module = WeightsResetV2(configuration=configuration)
    """

    def __init__(
        self, 
        *, 
        configuration: WeightsResetV2Configuration,
        module: nn.Module,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : WeightsResetV2Configuration
            The module configuration.
        module : nn.Module
            The target module.
        """

        super().__init__()

        self.weights_reset = WeightsReset(
            start=configuration.start,
            stop=configuration.stop,
            interval=configuration.interval,
            rate=configuration.rate,
            initializer=configuration.initializer,
            module=module,
        )
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.weights_reset(*args, **kwargs)
