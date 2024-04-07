from __future__ import annotations
from dataclasses import dataclass
from math import floor
from pprint import pprint

from osrsbox.monsters_api.monster_properties import MonsterProperties

from src.combat.model import Combatant, Mode, Stats
from osrsbox import monsters_api, items_api


_MOBS: dict[str, MonsterProperties] = {}


@dataclass(frozen=True)
class Mob(Combatant):
    name: str
    hp: int

    stats: Stats

    atk_mode: Mode
    atk_ival: float

    def_bonus: dict[Mode, int]
    atk_bonus: int
    str_bonus: int

    @classmethod
    def load(cls, name: str) -> Mob:
        global _MOBS
        if not _MOBS:
            _MOBS = {it.name.lower(): it for it in monsters_api.load()}

        raw_mob = _MOBS[name.lower()]

        stats = Stats(
            atk=raw_mob.attack_level,
            str=raw_mob.strength_level,
            dfc=raw_mob.defence_level,
            mag=raw_mob.magic_level,
            rng=raw_mob.ranged_level,
        )

        known_types = {m.name for m in Mode}
        attack_types = [
            t.upper() for t in raw_mob.attack_type if t.upper() in known_types
        ]

        if len(attack_types) != 1:
            raise NotImplementedError("Can't handle mobs with multiple attack types.")

        mode = Mode[attack_types[0].upper()]
        d_bonus = {
            Mode.STAB: raw_mob.defence_stab,
            Mode.SLASH: raw_mob.defence_slash,
            Mode.CRUSH: raw_mob.defence_crush,
            Mode.MAGIC: raw_mob.defence_magic,
            Mode.RANGED: raw_mob.defence_ranged,
        }

        return Mob(
            name=name,
            stats=stats,
            def_bonus=d_bonus,
            atk_mode=mode,
            hp=raw_mob.hitpoints,
            atk_ival=raw_mob.attack_speed * 0.6,
            # TODO factor out to class, share with loadout
            atk_bonus=raw_mob.attack_bonus,
            str_bonus=raw_mob.strength_bonus,
        )

    def atk_roll(self) -> int:
        return (self.stats.atk + 9) * (self.atk_bonus + 64)

    def def_roll(self, against: Mode) -> int:
        return (self.stats.dfc + 9) * (self.def_bonus[against] + 64)

    def max_hit(self) -> int:
        return floor(0.5 + (self.stats.str + 9) * (self.str_bonus + 64) / 640)


def load_mobs():
    all_mobs = monsters_api.load()
    for mob in all_mobs:
        pprint(mob.__dict__)
        break
    print(all_mobs)


def load_items():
    all_items = items_api.load()
    for item in all_items:
        if "toktz" in item.name.lower():
            pprint(item.__dict__)
            break


CRAB = Mob.load("ammonite crab")
BDRAG = Mob.load("bronze dragon")

if __name__ == "__main__":
    fg = Mob.load("Hill giant")
    print(fg.max_hit())
