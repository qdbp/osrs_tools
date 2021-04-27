from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Style(Enum):
    ACC = 1
    AGG = 2
    DEF = 3
    CNT = 4

    def a_bonus(self) -> int:
        if self == self.ACC:
            return 3
        elif self == self.CNT:
            return 1
        return 0

    def s_bonus(self) -> int:
        if self == self.AGG:
            return 3
        elif self == self.CNT:
            return 1
        return 0

    def d_bonus(self) -> int:
        if self == self.DEF:
            return 3
        elif self == self.CNT:
            return 1
        return 0


class Mode(Enum):
    STAB = 1
    SLASH = 2
    CRUSH = 3
    RANGED = 4
    MAGIC = 5


@dataclass()
class Equipment:
    name: str
    a_bonus: dict[Mode, int] = field(default_factory=dict)
    d_bonus: dict[Mode, int] = field(default_factory=dict)
    strength: int = 0
    prayer: int = 0

    @classmethod
    def dummy(cls):
        return Equipment(
            name="Dummy",
            a_bonus={m: 0 for m in Mode},
            strength=0,
        )


@dataclass()
class Weapon(Equipment):
    styles: dict[Style, Mode] = field(default_factory=dict)
    attack_ival: float = 2.4
    is_2h: bool = False

    @property
    def modes(self) -> list[Mode]:
        return [*self.styles.values()]

    def __post_init__(self):
        if not self.styles:
            raise ValueError(f"Weapon {self.name} must have at least one style!")

    def dominates(self, other: Weapon):
        """
        Returns:
            True if this weapon is at least as good as the other in every way.
            e.g. dscim >= rscim
        """
        return (
            self.is_2h <= other.is_2h
            and self.strength >= other.strength
            and all(s in self.styles for s in other.styles)
            and all(m in self.modes for m in other.modes)
            and all(self.a_bonus[m] >= other.a_bonus[m] for m in other.modes)
        )


@dataclass()
class Loadout:

    __slots__ = (
        "weapon",
        "shield",
        "head",
        "body",
        "legs",
        "boots",
        "gloves",
        "neck",
        "cape",
        "ring",
    )

    weapon: Weapon
    shield: Equipment

    head: Equipment
    body: Equipment
    legs: Equipment

    boots: Equipment
    gloves: Equipment

    neck: Equipment
    cape: Equipment
    ring: Equipment
