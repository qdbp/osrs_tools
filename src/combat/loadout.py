from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cache, cached_property
from itertools import product
from typing import Iterable

from src.combat.equipment import Equipment, Weapon
from src.combat.model import Mode, Stats


def eliminate_dominated(equipment: Iterable[Equipment]) -> list[Equipment]:
    """
    Naive algorithm, for now.
    """

    equipment = list(equipment)
    is_dominated = {i: False for i in equipment}

    for ex in range(len(equipment)):
        e = equipment[ex]
        if is_dominated[e]:
            continue

        for jx in range(ex + 1, len(equipment)):
            if equipment[jx].dominates(e):
                is_dominated[ex] = True

    return [k for k, dom in is_dominated.items() if not dom]


@dataclass(frozen=True)
class Loadout:
    weapon: Weapon = Weapon.dummy()
    shield: Equipment = Equipment.dummy()

    head: Equipment = Equipment.dummy()
    body: Equipment = Equipment.dummy()
    legs: Equipment = Equipment.dummy()

    boots: Equipment = Equipment.dummy()
    gloves: Equipment = Equipment.dummy()

    amulet: Equipment = Equipment.dummy()
    cape: Equipment = Equipment.dummy()
    ring: Equipment = Equipment.dummy()

    @classmethod
    def slots(cls) -> Iterable[str]:
        yield from [f.name for f in fields(cls)]

    @cached_property
    def prayer(self):
        return sum(it.prayer for it in self.iter_equipment())

    @cached_property
    def strength(self):
        return sum(it.strength for it in self.iter_equipment())

    @cache
    def get_a_bonus(self, mode: Mode) -> int:
        return sum(it.mode_a_bonus(mode) for it in self.iter_equipment())

    @cache
    def get_d_bonus(self, mode: Mode) -> int:
        return sum(it.mode_d_bonus(mode) for it in self.iter_equipment())

    def iter_equipment(self):
        yield self.weapon
        yield self.shield
        yield self.head
        yield self.body
        yield self.legs
        yield self.boots
        yield self.gloves
        yield self.amulet
        yield self.cape
        yield self.ring

    def __str__(self):
        return (
            f"Loadout["
            + "; ".join(
                f"{f.name}={eq_name}"
                for f in fields(self)
                if (eq_name := getattr(self, f.name).name) != "Dummy"
            )
            + "]"
        )

    __repr__ = __str__

    @cache
    def dominates(self, other: Loadout) -> bool:
        return (
            self.strength >= other.strength
            and self.prayer >= other.prayer
            and self.weapon.dominates(other.weapon)
            and all(self.get_a_bonus(m) >= other.get_a_bonus(m) for m in Mode)
            and all(self.get_d_bonus(m) >= other.get_d_bonus(m) for m in Mode)
        )

WEAPONS = [
    Weapon.load("Dragon scimitar"),
    Weapon.load("Abyssal whip"),
    Weapon.load("Abyssal dagger"),
    Weapon.load("Abyssal bludgeon"),
    Weapon.load("Verac's flail"),
    Weapon.load("Saradomin sword"),
]


def load_all(names: list[str]) -> list[Equipment]:
    return [Equipment.load(x) for x in names]


SHIELDS = load_all(["Unholy book", "Toktz-ket-xil", "dragon defender"])
HELMS = load_all(["Berserker helm", "Dragon med helm", "Rune full helm"])
BODIES = load_all(["Rune platebody", "Dragon chainbody"])
LEGS = load_all(["Rune platelegs", "Dragon platelegs"])
BOOTS = load_all(["Granite boots"])
GLOVES = load_all(["Combat bracelet", "Regen bracelet"])
AMULETS = load_all(["Amulet of fury"])
RINGS = load_all(["Brimstone ring"])
CAPES = load_all(["Obsidian cape"])


@cache
def get_candidate_loadouts(stats: Stats) -> tuple[Loadout, ...]:
    """
    Returns the set of all possible non-dominated loadouts for the given stats.
    """

    universe = {
        "weapon": WEAPONS,
        "shield": SHIELDS,
        "head": HELMS,
        "body": BODIES,
        "legs": LEGS,
        "boots": BOOTS,
        "gloves": GLOVES,
        "amulet": AMULETS,
        "cape": CAPES,
        "ring": RINGS,
    }
    assert set(universe.keys()) == {f.name for f in fields(Loadout)}

    filtered_universe = {
        k: eliminate_dominated([e for e in v if e.can_equip_at(stats)]) or [Equipment.dummy()]
        for k, v in universe.items()
    }

    if not filtered_universe["weapon"]:
        raise ValueError("All weapons eliminated, can't proceed. Stats might be too low.")

    # TODO this might get really really big, we might need stricter dominated filtering
    all_combos = product(*(filtered_universe.values()))

    return tuple(
        [
            Loadout(
                **dict(
                    zip(
                        universe.keys(),
                        combo if not combo[0].is_2h else (combo[0], Equipment.dummy(), *combo[2:]),
                    )
                )
            )
            for combo in all_combos
        ]
    )
