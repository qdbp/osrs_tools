from functools import lru_cache
from math import floor

import networkx as nx
import numpy as np
from numpy import cumsum

# TODO bonuses, etc.
from src.combat.gear import Equipment, Mode, Style, Weapon
from src.combat.mob import CRAB, Mob


def get_a_roll(
    base_level: int, style_bonus: int, equip_bonus: int, prayer_frac: float = 1.00
) -> int:
    el = base_level
    el = floor(el * prayer_frac)
    el += style_bonus + 8
    return el * (equip_bonus + 64)


def get_d_roll(base_level: int, style_bonus: int, equip_bonus: int) -> int:
    el = base_level + style_bonus + 8
    return el * (equip_bonus + 64)


def get_max_hit(
    base_level: int, style_bonus: int, equip_bonus: int, prayer_frac: float = 1.00
) -> int:
    el = base_level
    el = floor(prayer_frac * el)
    el += style_bonus + 8
    return floor(0.5 + el * (equip_bonus + 64) / 640)


def get_combat_stats(
    a_lvl: int,
    s_lvl: int,
    mob: Mob,
    style: Style,
    weapon: Weapon,
    equip: list[Equipment],
    s_pray: float = 1.00,
    a_pray: float = 1.00,
) -> tuple[float, float, int]:
    """
    Args:
        a_lvl:
        s_lvl:
        mob:
        style:
        weapon:
        equip:

    Returns:
        dps, hit_chance, max_hit

    """

    mode = weapon.styles[style]
    equip_a_bonus = weapon.a_bonus[mode] + sum(e.a_bonus[mode] for e in equip)
    equip_s_bonus = weapon.strength + sum(e.strength for e in equip)

    d_roll = get_d_roll(mob.l_def, 0, mob.d_bonus[mode])
    a_roll = get_a_roll(a_lvl, style.a_bonus(), equip_a_bonus, prayer_frac=a_pray)
    mh = get_max_hit(s_lvl, style.s_bonus(), equip_s_bonus, prayer_frac=s_pray)

    if a_roll > d_roll:
        hit_chance = 1 - (d_roll + 2) / (2 * a_roll + 2)
    else:
        hit_chance = 0.5 * a_roll / (d_roll + 1)

    return hit_chance * mh / (2 * weapon.attack_ival), hit_chance, mh


DSCIM = Weapon(
    "Dragon scimitar",
    a_bonus={Mode.SLASH: 67},
    strength=66,
    styles={Style.ACC: Mode.SLASH, Style.AGG: Mode.SLASH},
)
RSCIM = Weapon(
    "Rune scimitar",
    a_bonus={Mode.SLASH: 45},
    strength=44,
    styles={Style.ACC: Mode.SLASH, Style.AGG: Mode.SLASH},
)
BRSAB = Weapon(
    "Brine sabre",
    a_bonus={Mode.SLASH: 47},
    strength=46,
    styles={Style.ACC: Mode.SLASH, Style.AGG: Mode.SLASH},
)
GHAM = Weapon(
    "Granite hammer",
    a_bonus={Mode.CRUSH: 57},
    strength=56,
    styles={Style.ACC: Mode.CRUSH, Style.AGG: Mode.CRUSH},
)
AWHIP = Weapon(
    "Abyssal whip",
    a_bonus={Mode.SLASH: 82},
    strength=82,
    styles={Style.ACC: Mode.SLASH, Style.CNT: Mode.SLASH},
)
ADAGG = Weapon(
    "Abyssal dagger",
    a_bonus={Mode.STAB: 75},
    strength=75,
    styles={Style.ACC: Mode.STAB, Style.AGG: Mode.STAB},
)
ABLUD = Weapon(
    "Abyssal bludgeon",
    a_bonus={Mode.CRUSH: 102},
    strength=85,
    styles={Style.AGG: Mode.CRUSH},
)


XP_DIFFS = [
    floor(0.25 * floor(lvl - 1 + 300 * 2 ** ((lvl - 1) / 7))) for lvl in range(2, 100)
]
XP_VALS = [0] + np.cumsum(XP_DIFFS)
LVL_TABLE = {ix: i for ix, i in enumerate(cumsum(XP_DIFFS), 2)} | {1: 0}


# TODO make configurable
BASE_EQUIP = Equipment(
    # combat bracelet, glory amulet, berserker ring, unholy book
    name="base",
    a_bonus={
        Mode.SLASH: 29,
        Mode.STAB: 29,
        Mode.CRUSH: 29,
    },
    strength=21,
)


@lru_cache(maxsize=1 << 20)
def get_level(xp: int):
    for lvl, xv in enumerate(XP_VALS, 1):
        if xp < xv:
            return lvl
    return 99


def get_avail_weapons(a_lvl: int, s_lvl: int) -> list[Weapon]:

    out = []
    if a_lvl >= 40:
        out.append(RSCIM)
        out.append(BRSAB)
    if a_lvl >= 50:
        if s_lvl >= 50:
            out.append(GHAM)
    if a_lvl >= 60:
        out.append(DSCIM)
    if a_lvl >= 70:
        out.append(AWHIP)
        out.append(ADAGG)
        if s_lvl >= 70:
            out.append(ABLUD)

    return out


def xp_lattice(
    mob: Mob,
    base_equip: Equipment,
    a_start: int,
    s_start: int,
    strength_boost: int = 0,
    a_pray: float = 1.00,
    s_pray: float = 1.00,
    do_print: bool = True,
    xp_increment=25_000,
    target=(99, 99),
):

    g = nx.Graph()

    start_node = (LVL_TABLE[a_start], LVL_TABLE[s_start])
    nodes_to_process = {start_node}
    mh_table = {}
    hc_table = {}
    dps_table = {}
    w_table = {}
    xp_hr_table = {}

    while nodes_to_process:

        node = nodes_to_process.pop()
        g.add_node(node)

        # this gives the excess xp on top of the level xp
        axp, sxp = node
        al = get_level(axp)
        sl = get_level(sxp)

        # TODO handle cnt
        for style in [Style.ACC, Style.AGG, Style.CNT]:

            best_dps = 0
            for weapon in get_avail_weapons(al, sl):
                if style not in weapon.styles:
                    continue

                dps, hc, mh = get_combat_stats(
                    al,
                    sl + strength_boost,
                    mob,
                    style,
                    weapon,
                    [base_equip],
                    a_pray=a_pray,
                    s_pray=s_pray,
                )
                if dps > best_dps:
                    best_dps = dps
                    mh_table[node] = mh
                    hc_table[node] = hc
                    dps_table[node] = dps
                    w_table[node] = weapon
                    xp_hr_table[node] = 3600 * best_dps * 4
            if best_dps == 0:
                continue

            if style == Style.CNT:
                factor = 3600 * (4 / 3) * best_dps
            else:
                factor = 3600 * 4 * best_dps

            if style == Style.AGG and sl < 99:
                next_xp = LVL_TABLE[sl + 1]
                next_xp += xp_increment - next_xp % xp_increment
                delta = next_xp - sxp
                next_node = axp, sxp + delta
            # accurate
            elif style == Style.ACC and al < 99:
                next_xp = LVL_TABLE[al + 1]
                next_xp += xp_increment - next_xp % xp_increment
                delta = next_xp - axp
                next_node = axp + delta, sxp
            # controlled
            elif style == Style.CNT and al < 90 or sl < 90:
                next_axp = LVL_TABLE.get(al + 1, float("inf"))
                next_sxp = LVL_TABLE.get(sl + 1, float("inf"))

                if next_sxp < next_axp:
                    delta = next_sxp + xp_increment - next_sxp % xp_increment
                else:
                    delta = next_axp + xp_increment - next_axp % xp_increment

                next_node = axp + delta, sxp + delta
            else:
                continue

            g.add_node(next_node)
            g.add_edge(node, next_node, weight=delta / factor)
            nodes_to_process.add(next_node)

    # get node with the smallest xp reuquiremenet that meets our conditions
    target_node = sorted(
        [
            node
            for node in g.nodes
            if (target[0] is None or get_level(node[0]) == target[0])
            and (target[1] is None or get_level(node[1]) == target[1])
        ],
        key=lambda x: sum(x),
    )[0]

    cost = nx.algorithms.shortest_path_length(
        g, start_node, target_node, weight="weight"
    )
    path = nx.algorithms.shortest_path(g, start_node, target, weight="weight")

    if do_print:
        print(f"Training on {mob}, best target xp = {target_node}")
        print(f"Using equipment {base_equip}")
        print(f"Total time: {cost:.1f} hours.")
        print("Ideal strategy:")
        for u, v in zip(path[:-1], path[1:]):
            print(
                f"{get_level(u[0])},{get_level(u[1])} -> {get_level(v[0])},{get_level(v[1])}: "
                f"[time = {60 * g.edges[u, v]['weight']:.0f} min], "
                f"w/{w_table[u].name}, "
                f"mh {mh_table[u]}, "
                f"hc {hc_table[u]:.4%}, "
                f"dps {dps_table[u]:.4f} [{xp_hr_table[u]:.0f} xp/h]."
            )
    return cost


if __name__ == "__main__":
    cost = xp_lattice(CRAB, BASE_EQUIP, 62, 54, strength_boost=13, target=(99, 99))
    # cost = xp_lattice(CRAB, BASE_EQUIP, 60, 54, strength_boost=13)
