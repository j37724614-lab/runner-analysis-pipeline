"""管理需要暫時修改的程序級共享狀態。"""

import os
from contextlib import contextmanager
from threading import RLock

PROCESS_STATE_LOCK = RLock()


@contextmanager
def temporary_environment_variable(name: str, value: str):
    """暫時設定環境變數，離開區塊後恢復原值。"""
    was_set = name in os.environ
    original_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if was_set and original_value is not None:
            os.environ[name] = original_value
        else:
            os.environ.pop(name, None)
