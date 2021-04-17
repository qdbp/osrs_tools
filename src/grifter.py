from collections import Iterable
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.api import PriceAPI
from src.recipe import JEWEL_RECIPES, Recipe
import matplotlib.pyplot as plt


class Grifter:
    def __init__(self, api: PriceAPI):
        self.api = api

    def load_universe(
        self,
        window_days: int = 7,
        min_tot_volume: int = 10_000,
        min_lmt: int = 1,
        max_low: int = 1_000_000_000_000,
    ):

        cutoff = int(time()) - window_days * 86400

        with PriceAPI.db_conn() as conn:
            df = pd.read_sql(
                f"""
                SELECT prices.id, name, ts, high, low, hvol, lvol, lmt
                FROM prices INNER JOIN item_map on prices.id = item_map.id
                WHERE prices.ts >= {cutoff} AND item_map.lmt >= {min_lmt}
                """,
                conn,
                parse_dates={"ts": "s"},
            )

        df.fillna({"hvol": 0, "lvol": 0}, inplace=True)
        # FIXME resample to 5 minute intervals or otherwise
        df = df.groupby("id").filter(
            lambda df: (
                (df["lvol"].sum() >= min_tot_volume)
                and (df["low"].mean() <= max_low)
            )
        )
        return df

    def value_chain_snapshot_1h(self, recipe: Recipe):

        self.api.refresh_1h()

        cost = 0
        value = 0

        for name, count in recipe.ingredients:
            prices = api.prices_1h[api.id_from_name(name)]
            # be a little pessimistic
            cost += count * (2 / 3) * prices["avgHighPrice"] + (1 / 3) * prices[
                "avgLowPrice"
            ]
        for name, item in recipe.products:
            prices = api.prices_1h[api.id_from_name(name)]
            value += (1 / 3) * prices["avgHighPrice"] + (2 / 3) * prices[
                "avgLowPrice"
            ]

        print(f'Recipe {recipe.name}:')
        print(f'cost: {cost:.0f}')
        print(f'value: {cost:.0f}')
        print(f'roi: {(value - cost) / cost:.1%}')

    def scan_bands(self, smooth="30min"):

        df = self.load_universe(
            min_tot_volume=100_000,
            min_lmt=1000,
            max_low=30,
        )

        for item, idf in df.groupby("name"):

            idf = idf.rolling(smooth, on="ts").mean()

            min_price = idf["low"].min()
            max_price = idf["high"].max()

            prices = np.arange(min_price, max_price + 1, dtype=np.uint32)

            bands = {
                "low": [
                    len(idf[idf["low"].between(price - 0.5, price + 0.4999)])
                    / len(idf)
                    for price in prices
                ],
                "high": [
                    len(idf[idf["high"].between(price - 0.5, price + 0.4999)])
                    / len(idf)
                    for price in prices
                ],
            }

            crossover_rate = (idf["high"] < idf["low"] - 0.1).sum() / len(idf)

            band_df = DataFrame.from_dict(bands).set_index(prices)
            band_df["low_cum"] = band_df["low"].cumsum()
            band_df["high_cum"] = band_df["high"][::-1].cumsum()

            print(item)
            print(f"xover rate: {crossover_rate:.3f}")
            print(band_df)

    def scan_var(
        self,
        min_price: float = 5,
        max_price: float = 50,
        min_agg_vol: float = 25_000.0,
    ):

        df = self.load_universe(
            min_lmt=1000,
        )
        df.drop(["id", "high", "hvol"], axis=1, inplace=True)

        by_name = df.groupby("name")
        means = by_name[["low"]].mean()
        means = means.join(by_name[["lvol"]].sum())

        means.query(
            f"low <= {max_price} "
            f"& low >= {min_price} "
            f"& lvol >= {min_agg_vol}",
            inplace=True,
        )

        variation = by_name["low"].agg(lambda s: s.mad() / s.mean())
        variation.name = "variation"
        joint = means.join(variation).sort_values("variation")

        print(joint)


def plot_price_volume(
    names: Iterable[str],
    df: DataFrame,
    smooth: str = "30min",
):
    fig, (ax_p, ax_v) = plt.subplots(2, 1)

    for name in names:
        rolling = df.query(f'name == "{name}"').rolling(smooth, on="ts").mean()
        if len(rolling) == 0:
            print(f"{name} missing")
            continue

        rolling.plot(x="ts", y="high", ax=ax_p, label=name + "_high")
        rolling.plot(x="ts", y="low", ax=ax_p, label=name + "_low")

        rolling.loc[:, "pressure"] = rolling["lvol"] - rolling["hvol"]
        rolling.plot(x="ts", y="lvol", ax=ax_v, label=name + "_lvol")
        rolling.plot(x="ts", y="hvol", ax=ax_v, label=name + "_hvol")
        rolling.plot(x="ts", y="pressure", ax=ax_v, label=name + "_lmh")

    plt.show()
    return fig


if __name__ == "__main__":

    api = PriceAPI()
    grift = Grifter(api=api)

    for recipe in JEWEL_RECIPES:
        grift.value_chain_snapshot_1h(recipe)
