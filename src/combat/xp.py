from functools import lru_cache
from math import floor

import numpy as np
from numpy import cumsum

XP_DIFFS = [
    floor(0.25 * floor(lvl - 1 + 300 * 2 ** ((lvl - 1) / 7))) for lvl in range(2, 100)
]
XP_VALS = [0, 0] + list(np.cumsum(XP_DIFFS))
LVL_TABLE = {ix: i for ix, i in enumerate(cumsum(XP_DIFFS), 2)} | {1: 0}


@lru_cache(maxsize=1 << 20)
def get_level(xp: int):
    for lvl, xv in enumerate(XP_VALS, 1):
        if xp < xv:
            return lvl
    return 99


if __name__ == '__main__':
    print(XP_VALS)