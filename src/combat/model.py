from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from math import floor


class Combatant:
    @abstractmethod
    def atk_roll(self) -> int:
        pass

    @abstractmethod
    def def_roll(self, against: Mode) -> int:
        pass

    @abstractmethod
    def max_hit(self) -> int:
        pass


@dataclass(frozen=True)
class Stats:
    atk: int = 1
    str: int = 1
    dfc: int = 1
    mag: int = 1
    rng: int = 1

    def __str__(self):
        return f"A{self.atk} S{self.str} D{self.dfc} R{self.rng} M{self.mag}"

    __repr__ = __str__


class Style(Enum):
    ACC = 1
    AGG = 2
    DEF = 3
    CON = 4
    MAG = 5
    MAG_DEF = 6
    RNG_RAP = 7
    RNG_LNG = 8
    RNG_ACC = 9

    @property
    def a_bonus(self) -> int:
        if self == self.ACC:
            return 3
        elif self == self.CON:
            return 1
        return 0

    @property
    def s_bonus(self) -> int:
        if self == self.AGG:
            return 3
        elif self == self.CON:
            return 1
        return 0

    @property
    def d_bonus(self) -> int:
        if self == self.DEF:
            return 3
        elif self == self.CON:
            return 1
        return 0

    @classmethod
    @lru_cache()
    def from_osrsbox(cls, key: str) -> Style:
        return {
            "accurate": Style.ACC,
            "defensive": Style.DEF,
            "aggressive": Style.AGG,
            "controlled": Style.CON,
            "spellcasting": Style.MAG,
            "defensive casting": Style.MAG_DEF,
        }[key]

    def __str__(self):
        return self.name

    __repr__ = __str__


class Mode(Enum):
    STAB = 1
    SLASH = 2
    CRUSH = 3
    RANGED = 4
    MAGIC = 5

    @classmethod
    @lru_cache()
    def from_osrsbox(cls, key: str) -> Mode:
        return {
            "stab": Mode.STAB,
            "slash": Mode.SLASH,
            "crush": Mode.CRUSH,
            "ranged": Mode.RANGED,
            "magic": Mode.MAGIC,
        }[key]

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return self.name

    __repr__ = __str__


def atk_roll(base_level: int, style_bonus: int, equip_bonus: int, prayer_frac: float = 1.00) -> int:
    effective_level = base_level
    effective_level = floor(effective_level * prayer_frac)
    effective_level += style_bonus + 8
    return effective_level * (equip_bonus + 64)


def def_roll(base_level: int, style_bonus: int, equip_bonus: int) -> int:
    effective_level = base_level + style_bonus + 8
    return effective_level * (equip_bonus + 64)


def get_max_hit(
    base_strength: int, style_bonus: int, equip_bonus: int, prayer_frac: float = 1.00
) -> int:
    effective_strength = base_strength
    effective_strength = floor(prayer_frac * effective_strength)
    effective_strength += style_bonus + 8
    return floor(0.5 + effective_strength * (equip_bonus + 64) / 640)
