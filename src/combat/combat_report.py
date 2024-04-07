from __future__ import annotations

from dataclasses import dataclass

from math import floor

from src.combat.loadout import Loadout
from src.combat.mob import Mob
from src.combat.model import Stats, Style, atk_roll, def_roll, get_max_hit


@dataclass(frozen=True)
class CombatReport:
    """
    Summarizes the relevant statistics of head-to-head combat between a player and a mob.
    """

    loadout: Loadout
    against: Mob
    style: Style

    hit_chance: float
    max_hit: int
    dps: float

    back_dps: float

    def kph(self, turnover_time: float = 2.0):
        """
        Kills per hour, given a particular turnover time.
        """
        return 3600 / (self.against.hp / self.dps + turnover_time)

    def xpph(self, turnover_time: float = 2.0):
        """
        XP per hour, given a particular turnover time.
        """
        return 4 * self.kph(turnover_time) * self.against.hp

    # TODO implement
    def sustain(self, food: int = 13 * 20) -> float:
        """
        How long we can last in continuous combat, given a particular food supply.

        Will be infinite if we can sustain indefinitely.
        """
        raise NotImplementedError()

    @classmethod
    def calculate(
        cls,
        stats: Stats,
        mob: Mob,
        style: Style,
        loadout: Loadout,
        black_mask: bool = False,
        a_boost: int = 0,
        s_boost: int = 0,
        d_boost: int = 0,
        s_pray: float = 1.00,
        a_pray: float = 1.00,
        d_pray: float = 1.00,
    ) -> CombatReport:
        """
        Args:
            stats:
            loadout:
            mob:
            style: attack style to use
            s_pray: strength prayer, as net multiplier; e.g. for +10%, this should be 1.10
            a_pray: attack prayer, as net multiplier; e.g. for +10%, this should be 1.10

        Returns:
            a combat report object

        """

        mode = loadout.weapon.modes_by_style[style]

        d_roll = def_roll(mob.stats.dfc, 1, mob.def_bonus[mode])
        a_roll = atk_roll(
            stats.atk + a_boost, style.a_bonus, loadout.get_a_bonus(mode), prayer_frac=a_pray
        )

        max_hit = get_max_hit(stats.str + s_boost, style.s_bonus, loadout.strength, prayer_frac=s_pray)

        if black_mask:
            a_roll = floor(7 * a_roll / 6)
            max_hit = floor(7 * max_hit / 6)

        if not loadout.weapon.name.startswith("osmumten"):
            hit_chance = get_hit_chance(a_roll, d_roll)
        else:
            hit_chance = get_hit_chance_osm(a_roll, d_roll)
        dps = hit_chance * max_hit / (2 * loadout.weapon.attack_ival)

        back_a_roll = atk_roll(mob.stats.atk, 1, mob.atk_bonus)
        back_d_roll = def_roll(stats.dfc + d_boost, style.d_bonus, loadout.get_d_bonus(mob.atk_mode))
        back_mh = get_max_hit(mob.stats.str, 1, 0)

        back_hc = get_hit_chance(back_a_roll, back_d_roll)
        back_dps = back_hc * back_mh / (2 * mob.atk_ival)

        return CombatReport(
            loadout=loadout,
            against=mob,
            hit_chance=hit_chance,
            max_hit=max_hit,
            dps=dps,
            back_dps=back_dps,
            style=style,
        )


def get_hit_chance(a_roll: int, d_roll: int) -> float:
    if a_roll > d_roll:
        return 1 - (d_roll + 2) / (2 * a_roll + 2)
    else:
        return 0.5 * a_roll / (d_roll + 1)


def get_hit_chance_osm(a_roll, d_roll):
    if a_roll >= d_roll:
        return 1 - (d_roll + 2) * (2 * d_roll + 3) / (6 * (a_roll + 1) ** 2)
    else:
        return a_roll * (4 * a_roll + 5) / (6 * (a_roll + 1) * (d_roll + 1))
