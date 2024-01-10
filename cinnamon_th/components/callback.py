from typing import Optional, Dict

import numpy as np
from copy import deepcopy
import torch as th

from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.callback import Callback


class THEarlyStopping(Callback):
    """
    Early stopping callback.
    Mainly inspired from the Keras implementation.
    """

    def __init__(
            self,
            **kwargs
    ):
        super(THEarlyStopping, self).__init__(**kwargs)

        self.wait: int = 0
        self.best_epoch: int = 0
        self.stopped_epoch: int = 0
        self.best_value: Optional[float] = None
        self.best_weights: Optional[int] = None
        self.has_restored: bool = False

        if self.mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best_value = np.inf
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best_value = -np.inf

    def reset(
            self
    ):
        """
        Resets the ``EarlyStopping`` internal state for component re-use.
        """

        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.has_restored = False
        self.best_weights = None
        if self.baseline is not None:
            self.best_value = self.baseline
        else:
            self.best_value = np.Inf if self.monitor_op == np.less else -np.Inf

        self.component.model.stop_training = False

    def on_fit_begin(
            self,
            logs: Optional[Dict] = None
    ):
        # Allow instances to be re-used
        self.reset()

    def restore_weights(
            self
    ):
        with th.no_grad():
            self.component.model.load_state_dict(self.best_weights)

    def on_epoch_end(
            self,
            logs: Optional[Dict] = None
    ):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.best_epoch = logs['epoch']
            self.wait = 0
            if self.restore_best_weights:
                with th.no_grad():
                    self.best_weights = deepcopy(self.component.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = logs['epoch']
                self.component.model.stop_training = True
                if self.restore_best_weights:
                    logging_utility.logger.info(f'Restoring model weights from the end of the best epoch ({self.best_epoch}).')
                    self.restore_weights()
                    self.has_restored = True

    def on_fit_end(
            self,
            logs: Optional[Dict] = None
    ):
        if self.stopped_epoch > 0:
            logging_utility.logger.info(f'Early stopping best epoch: {self.best_epoch}')

        # Done when maximum epochs is reached and early stopping has not triggered
        if self.restore_best_weights and not self.has_restored:
            logging_utility.logger.info(f'Restoring model weights from the end of the best epoch: {self.best_epoch}')
            self.restore_weights()

        self.reset()

    def get_monitor_value(
            self,
            logs: Optional[Dict] = None
    ):
        """
        Retrieves the early stopping metric value (i.e., `monitor`) based on the given configuration.
        This method is invoked at the end of each epoch to decide whether to perform early stopping or not.

        Args:
            logs: A dictionary containing callback hookpoint information.

        Returns:
            The early stopping metric value
        """

        logs = logs if logs is not None else {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging_utility.logger.warning(
                f'Early stopping conditioned on metric {self.monitor} which is not available.'
                f' Available metrics are: {",".join(list(logs.keys()))}')
        return monitor_value
