from dataclasses import dataclass
from datetime import datetime
from functools import partial
from time import sleep
from typing import Iterable, NoReturn
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from colored import fg, stylize
from obspy.signal.detrend import polynomial as poly_detrend
from pandas import DataFrame, Timedelta
from scipy.stats import linregress

from src.merch.api import PriceAPI
from src.merch.recipe import BankableRecipe, Item, Recipe

filterwarnings("error", message="Mean of empty slice")


@dataclass
class Grifter:
    api: PriceAPI

    def value_chain_snapshot_1h(self, recipe: Recipe) -> None:

        self.api.refresh_1h()

        cost = 0
        value = 0

        for name, count in recipe.ingredients.items():
            prices = api.prices_1h[name]
            # be a little pessimistic
            cost += count * (2 / 3) * prices["avgHighPrice"] + (1 / 3) * prices["avgLowPrice"]

        for name, count in recipe.products.items():
            prices = api.prices_1h[name]
            value += (count / 3) * prices["avgHighPrice"] + (2 * count / 3) * prices["avgLowPrice"]

        profit = value - cost

        print(f"Recipe {recipe.name}: {recipe.ingredients} -> {recipe.products}")
        print(f"\t{value: 5.0f} - {cost: 5.0f} => {profit: 5.0f} ({profit / cost:.1%} roi)")

        if isinstance(recipe, BankableRecipe):
            buy_amt = float("inf")
            for name, count in recipe.ingredients.items():
                buy_amt = min(buy_amt, api.prices_1h[name]["lowPriceVolume"] / (3 * count))

            print(f"\tAssuming we capture 1/3 the volume, we can make {buy_amt:.1f} / hour")
            print(f"\tThis will make ${buy_amt * profit:.0f}")
            print(
                f"\tAssuming we work nonstop, we can make " f"${profit * recipe.pps(buy_amt):.0f} / hour"
            )

    def scan_var(
        self,
        min_price: float = 5,
        max_price: float = 10_000,
        min_4h_vol: float = 2500,
        max_null_ratio: float = 0.3,
        min_limit: int = 200,
        quantile_days: int = 1,
    ):
        """
        Runs a variance-based scan to try and find markets in need of a market-maker.

        Args:
            min_price:
            max_price:
            min_4h_vol:
            max_null_ratio:
            detrend_order:
            jitter_lookback_days:
            max_trend:
            min_jitter:
            min_profit:
            min_roi:
            min_limit:

        Returns:

        """

        prices, meta = self.api.load_universe(
            min_lmt=min_limit,
            min_low=min_price,
            max_low=max_price,
            min_4h_vol=min_4h_vol,
            max_null_ratio=max_null_ratio,
            quantile_days=1,
        )
        prices.drop(["id"], axis=1, inplace=True)

        def get_nd_trend(n: int, idf: DataFrame):
            name = idf.name
            idf = idf[idf["time"] >= idf["time"].max() - Timedelta(days=n)]
            x, y = idf["time"].astype(int) / (1e9 * 86400), idf["low"].values
            x -= x.iloc[0]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return 100 * slope / n / meta.loc[name, "q50_low"]

        meta["q50_mid"] = 0.5 * (meta["q50_low"] + meta["q50_high"])
        meta["trend_%/d_7d"] = prices.groupby("name").apply(partial(get_nd_trend, 7))
        meta["trend_%/d_30d"] = prices.groupby("name").apply(partial(get_nd_trend, 30))

        meta["margin"] = meta["q80_high"] - meta["q20_low"] - 0.01 * meta["q50_low"]
        meta["flow"] = np.minimum(0.15 * meta["4h_vol"], meta["limit"])
        meta["score"] = meta["flow"] * meta["margin"]
        meta["capex"] = meta["q50_low"] * meta["flow"]
        meta["roi"] = meta["score"] / meta["capex"]

        meta_clean = meta.drop(list(meta.filter(regex="^q")), axis=1)
        meta_clean["q50"] = meta["q50_mid"]
        meta_clean.sort_values("score", ascending=False, inplace=True)
        meta_clean = meta_clean[
            (meta_clean["trend_%/d_7d"].abs() < 0.35) & (meta_clean["trend_%/d_30d"].abs() < 0.35)
        ]
        meta_clean = meta_clean[(meta_clean["capex"] >= 3000000)]
        meta_clean = meta_clean[meta_clean["flow"] >= 1000]

        print(meta_clean.sort_values("score"))

    def scan_ha(self, min_profit: int = 500):
        @dataclass()
        class HAProfit:
            item: Item
            cost: int

            @property
            def profit(self) -> int:
                return int(0.6 * self.item.value - self.cost)

            @property
            def total_profit(self) -> int:
                return self.item.limit * self.profit

            def __str__(self):
                return (
                    f"HA for {self.item.name}/{self.cost}: {self.profit} "
                    f"(x{self.item.limit} = {self.total_profit})"
                )

        self.api.refresh_1h()
        prices = self.api.prices_1h

        nat = self.api.by_name["Nature rune"]
        nat_cost = prices[nat.name]["avgHighPrice"]
        profits = []

        for name, item in self.api.by_name.items():
            if not 1000 <= item.value <= 100_000:
                continue
            if item.name not in prices:
                continue

            high_cost = prices[item.name]["avgHighPrice"]
            low_cost = prices[item.name]["avgLowPrice"]
            if high_cost is None or low_cost is None:
                continue

            base_cost = 0.5 * low_cost + 0.5 * high_cost
            ha_record = HAProfit(item, base_cost + nat_cost)

            if ha_record.profit > min_profit:
                profits.append(ha_record)

        for record in sorted(profits, key=lambda rec: -(rec.profit)):
            print(record)

    def get_spectra(self, items: Iterable[str]):

        df, meta = self.api.load_universe()

        fig, ax = plt.subplots()

        ax.set_xlabel("frequency, per-hour")
        ax.set_ylabel("spectral price power")
        ax.set_ylim(0, 100)

        for item in items:
            try:
                low = df.loc[item, "low"]
            except KeyError:
                print(f"{item} not in universe!")
                continue

            l, u = np.quantile(low, [0.03, 0.97])
            low = low.clip(l, u)
            poly_detrend(low.values, order=3)

            # 3-day window, assumes 5 min samples (e.g. 12 per hour)
            freq, power = sg.periodogram(low, fs=12, nfft=4 * 12)
            print(freq)
            ax.plot(freq, power, label=item)

        ax.legend()

        plt.show()

    def scan_drop(self, min_drop: float = 0.8) -> NoReturn:

        _, meta = self.api.load_universe(window_days=7, max_null_ratio=0.3, min_4h_vol=5000, min_low=50)

        cur_lows = {}
        new_lows = {}

        while True:
            self.api.refresh_1h()
            self.api.refresh_5m()

            out = self.api.latest()
            new_lows.clear()

            for name in meta.index:

                high = out[name]["high"]

                base_q50 = meta.loc[name, "q50_low"]
                p_q50 = (high + 10) / (base_q50 + 10)

                try:
                    base_1h = self.api.prices_1h[name]["avgLowPrice"]
                    p_1h = (high + 10) / (base_1h + 10)
                except (KeyError, TypeError):
                    base_1h = base_q50
                    p_1h = float("nan")
                try:
                    base_5m = self.api.prices_5m[name]["avgLowPrice"]
                    p_5m = (high + 10) / (base_5m + 10)
                except (TypeError, KeyError):
                    base_5m = base_q50
                    p_5m = float("nan")

                drop = min(p_q50, p_1h, p_5m)
                base = min([base_5m, base_1h, base_q50])

                if drop < min_drop or base - high > 150:
                    new_lows[name] = high, base, drop

            for name, (high, base, drop) in new_lows.items():
                if name not in cur_lows:
                    print(f"{datetime.now()}: {name}: {high=:.0f}, {base=:.0f}, {drop=:.3f}")

            cur_lows.clear()
            cur_lows |= new_lows
            sleep(30)

    def live_feed(self, items: Iterable[str]):

        orange = fg("orange_3")
        green = fg("green")

        items = list(items)

        last_high = {}
        last_low = {}

        while True:
            for item in items:
                latest = self.api.latest(item)

                ht, hp = latest["highTime"], latest["high"]
                lt, lp = latest["lowTime"], latest["low"]

                if hp != last_high.get(item):
                    last_high[item] = hp
                    msg = stylize(f'{ht} [{item.upper()}] H: {latest["high"]}', orange)
                    print(msg)
                if lp != last_low.get(item):
                    last_low[item] = lp
                    msg = stylize(f'{ht} [{item.upper()}] L: {latest["low"]}', green)
                    print(msg)

            sleep(5)


# TODO scanner: mad(1d) / mad(3d) [recent price instability]


def plot_price_volume(names: Iterable[str], df: DataFrame, smooth: str = "30min"):
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
    api = PriceAPI(refresh=False)
    grift = Grifter(api=api)

    # crft = 56
    # gem_recipes = [get_opal_recipe(crft), get_jade_recipe(crft), get_rtopaz_recipe(crft)]

    # for recipe in gem_recipes + JEWEL_RECIPES:
    #     grift.value_chain_snapshot_1h(recipe)

    grift.scan_var(min_price=100, max_price=200_000, min_4h_vol=500, min_limit=200)
