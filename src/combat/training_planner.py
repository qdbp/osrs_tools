from dataclasses import dataclass, replace

import networkx as nx
from tqdm import tqdm

from src.combat.combat_report import CombatReport
from src.combat.loadout import get_candidate_loadouts
from src.combat.mob import Mob
# TODO bonuses, etc.
from src.combat.model import Stats, Style
from src.combat.xp import XP_VALS


@dataclass(frozen=True)
class TrainingEnemy:
    mob: Mob
    turnover: float


def calculate_training_path(
    enemy: TrainingEnemy,
    start_stats: Stats,
    strength_boost: int = 0,
    attack_boost: int = 0,
    a_pray: float = 1.00,
    s_pray: float = 1.00,
    do_print: bool = True,
    target_stats=Stats(99, 99, 99),
):
    g = nx.Graph()
    nodes_to_process = {start_stats}

    mh_table = {}
    hc_table = {}
    dps_table = {}
    loadout_table = {}
    style_table = {}
    xp_hr_table = {}

    prog = tqdm()
    while nodes_to_process:

        node = nodes_to_process.pop()

        prog.update()
        prog.set_description(
            f"Nodes to process: {len(nodes_to_process)} ({node.atk + node.str + node.dfc})"
        )

        # TODO handle cnt
        for style in [Style.ACC, Style.AGG, Style.DEF]:
            best_cr = None
            for loadout in get_candidate_loadouts(node):
                if style not in loadout.weapon.modes_by_style.keys():
                    continue
                cr = CombatReport.calculate(
                    node,
                    enemy.mob,
                    style,
                    loadout,
                    s_boost=strength_boost,
                    a_boost=attack_boost,
                    a_pray=a_pray,
                    s_pray=s_pray,
                )
                if best_cr is None or cr.dps > best_cr.dps:
                    best_cr = cr
                    mh_table[node, style] = cr.max_hit
                    hc_table[node, style] = cr.hit_chance
                    dps_table[node, style] = cr.dps
                    loadout_table[node, style] = loadout
                    xp_hr_table[node, style] = best_cr.xpph(enemy.turnover)

            if best_cr is None:
                continue

            if style == Style.AGG and node.str < target_stats.str:
                next_node = replace(node, str=node.str + 1)
                xp_delta = XP_VALS[node.str + 1] - XP_VALS[node.str]
            # accurate
            elif style == Style.ACC and node.atk < target_stats.atk:
                next_node = replace(node, atk=node.atk + 1)
                xp_delta = XP_VALS[node.atk + 1] - XP_VALS[node.atk]
            # defensive
            elif style == Style.DEF and node.dfc < target_stats.dfc:
                next_node = replace(node, dfc=node.dfc + 1)
                xp_delta = XP_VALS[node.dfc + 1] - XP_VALS[node.dfc]
            # TODO make this not painfully slow
            # elif style == Style.CON and al < tgt_a_lvl or sl < tgt_s_lvl or dl < tgt_d_lvl:
            #     da = LVL_TABLE.get(al + 1, float("inf")) - axp
            #     ds = LVL_TABLE.get(sl + 1, float("inf")) - sxp
            #     dd = LVL_TABLE.get(dl + 1, float("inf")) - dxp
            #     delta = r_up(min(da, ds, dd))
            #     next_node = axp + delta, sxp + delta, dxp + delta
            else:
                continue

            edge_cost_hours = xp_delta / xp_hr_table[node, style]

            if not g.has_node(next_node):
                g.add_node(next_node)
                nodes_to_process.add(next_node)

            if not g.has_edge(node, next_node) or g.edges[node, next_node]["weight"] > edge_cost_hours:
                g.add_edge(node, next_node, weight=edge_cost_hours)
                style_table[node, next_node] = style

    # get node with the smallest xp reuquiremenet that meets our conditions
    print(f"finding shortest for target {target_stats}")

    xp_delta = nx.algorithms.shortest_path_length(g, start_stats, target_stats, weight="weight")
    path = nx.algorithms.shortest_path(g, start_stats, target_stats, weight="weight")

    if do_print:
        print(f"Training on {enemy.mob.name} until {target_stats}")
        print(f"Total time: {xp_delta:.1f} hours.")
        print("Ideal strategy:")
        u: Stats
        v: Stats
        for u, v in zip(path[:-1], path[1:]):
            style = style_table[u, v]
            ld = loadout_table[u, style]
            print(
                f"{u} -> {v}:"
                f"{ld.weapon}({style_table[u, v]}) "
                f"[time = {60 * g.edges[u, v]['weight']:.0f} min], "
                f"mh {mh_table[u, style]}, "
                f"hc {hc_table[u, style]:.4%}, "
                f"dps {dps_table[u, style]:.4f} [{xp_hr_table[u, style]:.0f} xp/h]"
                f"\n\t-> {ld}, "
            )
    return xp_delta


if __name__ == "__main__":
    stats = Stats(73, 68, 67)
    # lds = get_candidate_loadouts(stats)

    enemy = TrainingEnemy(Mob.load("Ammonite crab"), 1.0)
    # enemy, turnover = Mob.load("Ammonite crab"), 1.0
    calculate_training_path(enemy, stats, target_stats=Stats(99, 99, 67))

    # reports = []
    # for ld in lds:
    #     for style in ld.weapon.modes_by_style.keys():
    #         r = get_combat_report(stats, enemy, loadout=ld, style=style, s_boost=0)
    #         reports.append(r)

    # for r in sorted(reports, key=lambda x: x.dps)[::-1][:5]:
    #     print(r.loadout)
    #     print(f'{r.loadout.weapon}@{r.style}, dps={r.dps:.3f}, mh={r.max_hit}')
