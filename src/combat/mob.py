from dataclasses import dataclass

from src.combat.gear import Mode


@dataclass()
class Mob:
    l_att: int
    l_str: int
    l_def: int

    d_bonus: dict[Mode, int]


CRAB = Mob(1, 1, 1, {m: 0 for m in Mode})
HGIANT = Mob(18, 22, 26, {m: 0 for m in Mode})
MGIANT = Mob(30, 30, 30, {m: 0 for m in Mode})