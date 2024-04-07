from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any, Type, TypeVar

from osrsbox import items_api

from src.combat.model import Mode, Stats, Style

Eq = TypeVar("Eq", bound="Equipment")


class Feature:
    pass


_EQUIPMENT = {it.name.lower(): it for it in items_api.load() if it.equipment is not None}


@dataclass(frozen=True)
class Equipment:
    name: str
    a_bonus: dict[Mode, int] = field(default_factory=dict)
    d_bonus: dict[Mode, int] = field(default_factory=dict)

    strength: int = 0
    prayer: int = 0

    req: Stats = Stats()
    features: tuple[Feature] = ()

    @classmethod
    def dummy(cls):
        return cls(name="Dummy")

    def mode_d_bonus(self, mode: Mode):
        return self.d_bonus.get(mode, 0)

    def mode_a_bonus(self, mode: Mode):
        return self.a_bonus.get(mode, 0)

    @cache
    def dominates(self, other: Equipment):
        """
        Returns:
            True if this equipment is at least as good as the other in
            every possible way.
        """
        return (
            type(self) is type(other)
            and self.strength >= other.strength
            and not (set(other.features) - set(self.features))
            and all(self.a_bonus.get(m, 0) >= other.a_bonus.get(m, 0) for m in other.a_bonus.keys())
            and all(self.d_bonus.get(m, 0) >= other.d_bonus.get(m, 0) for m in other.d_bonus.keys())
        )

    def can_equip_at(self, stats: Stats) -> bool:
        return self.req.atk <= stats.atk and self.req.dfc <= stats.dfc and self.req.str <= stats.str

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name.upper()

    __repr__ = __str__

    @classmethod
    def deserialize(cls, name: str) -> dict[str, Any]:
        raw_eq = _EQUIPMENT[name.lower()].equipment
        raw_req = raw_eq.requirements or {}

        kwargs = {
            "name": name,
            "d_bonus": {
                Mode.STAB: raw_eq.defence_stab,
                Mode.SLASH: raw_eq.defence_slash,
                Mode.CRUSH: raw_eq.defence_crush,
                Mode.MAGIC: raw_eq.defence_magic,
                Mode.RANGED: raw_eq.defence_ranged,
            },
            "a_bonus": {
                Mode.STAB: raw_eq.attack_stab,
                Mode.SLASH: raw_eq.attack_slash,
                Mode.CRUSH: raw_eq.attack_crush,
                Mode.MAGIC: raw_eq.attack_magic,
                Mode.RANGED: raw_eq.attack_ranged,
            },
            "strength": raw_eq.melee_strength,
            "prayer": raw_eq.prayer,
            "req": Stats(
                atk=raw_req.get("attack", 1),
                str=raw_req.get("strength", 1),
                dfc=raw_req.get("defence", 1),
                rng=raw_req.get("ranged", 1),
                mag=raw_req.get("magic", 1),
            )
            # TODO feature handling
        }

        return kwargs

    @classmethod
    def load(cls: Type[Eq], name: str) -> Eq:
        return cls(**cls.deserialize(name))


@dataclass(frozen=True)
class Weapon(Equipment):
    modes_by_style: dict[Style, Mode] = field(default_factory=dict)
    attack_ival: float = 2.4
    is_2h: bool = False

    @classmethod
    def dummy(cls):
        return Weapon("Dummy", modes_by_style={Style.ACC: Mode.SLASH})

    @property
    def attack_modes(self) -> list[Mode]:
        return [*self.modes_by_style.values()]

    def __post_init__(self):
        if not self.modes_by_style:
            raise ValueError(f"Weapon {self.name} must have at least one style!")

    @classmethod
    def deserialize(cls, name: str) -> dict[str, Any]:
        base = super().deserialize(name)

        raw_eq = _EQUIPMENT[name.lower()]
        raw_weapon = raw_eq.weapon

        # need special handling to handle osrsbox mixing of styles/types
        modes_by_style = {}
        for stance in raw_weapon.stances:
            box_type = stance["attack_type"]
            box_style = stance["attack_style"]

            if box_style == "magic":
                mode = Mode.MAGIC
                style = Style.from_osrsbox(box_type)
            else:
                mode = Mode.from_osrsbox(box_type)
                style = Style.from_osrsbox(box_style)

            modes_by_style[style] = mode

        base |= {
            "attack_ival": raw_weapon.attack_speed * 0.6,
            "is_2h": raw_eq.equipment.slot == "2h",
            "modes_by_style": modes_by_style,
        }
        return base

    @cache
    def dominates(self, other: Weapon):
        return Equipment.dominates(self, other) and (
            self.is_2h <= other.is_2h
            and all(s in self.modes_by_style for s in other.modes_by_style)
            and all(m in self.attack_modes for m in other.attack_modes)
        )

    def __hash__(self):
        return self.name.__hash__()


if __name__ == "__main__":
    test = Weapon.load("rune 2h sword")
    test = Weapon.load("abyssal whip")

    print(test)
