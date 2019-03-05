__author__ = "liuwei"

"""
write the loss and accu to the tensorboardx
"""
from typing import Any
from tensorboardX import SummaryWriter

class TensorboardWriter:
    """
    wrap the SummaryWriter, print the value to the tensorboard
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None,
                 test_log: SummaryWriter = None):
        self._train_log = train_log
        self._validation_log = validation_log
        self._test_log = test_log

    @staticmethod
    def _item(value: Any):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value

        return val

    def add_train_scalar(self, name: str, value: float, global_step: int):
        """
        add train scalar value to tensorboardX
        Args:
            name: the name of the value
            value:
            global_step: the steps
        """
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)


    def add_validation_scalar(self, name: str, value: float, global_step: int):
        """
        add validation scalar value to tensorboardX
        Args:
            name: the name of the value
            value:
            global_step:
        """
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), global_step)

    def add_test_scalar(self, name: str, value: float, global_step: int):
        """
        add test scalar value to tensorboardX
        Args:
            name: the name of the value
            value:
            global_step:
        """
        if self._test_log is not None:
            self._test_log.add_scalar(name, self._item(value), global_step)

