from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Iterable
from warnings import filterwarnings
from colored import fg, stylize

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from obspy.signal.detrend import polynomial as poly_detrend
from pandas import DataFrame, Series

from src.api import PriceAPI
from src.recipe import Item, Recipe

filterwarnings("error", message="Mean of empty slice")


class Grifter:
    def __init__(self, api: PriceAPI):
        self.api = api

    def value_chain_snapshot_1h(self, recipe: Recipe):

        self.api.refresh_1h()

        cost = 0
        value = 0

        for name, count in recipe.ingredients.items():
            prices = api.prices_1h[name]
            # be a little pessimistic
            cost += (
                count * (2 / 3) * prices["avgHighPrice"]
                + (1 / 3) * prices["avgLowPrice"]
            )

        for name, item in recipe.products.items():
            prices = api.prices_1h[name]
            value += (1 / 3) * prices["avgHighPrice"] + (2 / 3) * prices["avgLowPrice"]

        print(f"Recipe {recipe.name}:")
        print("\t", recipe.ingredients)
        print(f"\tcost: {cost:.0f}")
        print(f"\tvalue: {value:.0f}")
        print(f"\tprofit: {value - cost:.0f}")
        print(f"\troi: {(value - cost) / cost:.1%}")

    def scan_bands(self, smooth="30min"):

        prices, meta = self.api.load_universe(
            min_4h_vol=100_000,
            min_lmt=1000,
            max_low=30,
        )

        for item, idf in prices.groupby("name"):

            idf = idf.rolling(smooth, on="ts").mean()

            min_price = idf["low"].min()
            max_price = idf["high"].max()

            prices = np.arange(min_price, max_price + 1, dtype=np.uint32)

            bands = {
                "low": [
                    len(idf[idf["low"].between(price - 0.5, price + 0.4999)]) / len(idf)
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
        max_price: float = 10_000,
        min_4h_vol: float = 2500,
        max_null_ratio: float = 0.3,
        detrend_order=7,
        jitter_lookback_days: int = 5,
        max_trend: float = 0.015,
        min_jitter: float = 0.03,
        min_profit: float = 30_000,
        min_roi: float = 1.0,
    ):

        prices, meta = self.api.load_universe(
            min_lmt=1000, min_low=min_price, max_low=max_price, min_4h_vol=min_4h_vol
        )
        prices.drop(["id", "high", "hvol"], axis=1, inplace=True)

        ts = prices.index.get_level_values(1)
        span_secs = (ts.max() - ts.min()).total_seconds()

        # number of 4-hour spans
        n_spans = span_secs / (86400 / 6)

        by_name = prices.groupby("name")

        means = by_name[["low"]].mean()
        means = means.join(by_name["lvol"].agg(lambda s: s.sum() / n_spans))
        means = means.join(meta)
        means.query(
            (
                f"low <= {max_price} & low >= {min_price}"
                f"& lvol >= {min_4h_vol}"
                f"& null_ratio <= {max_null_ratio}"
            ),
            inplace=True,
        )

        def jitter(s: Series) -> float:
            trunc = s.values[-jitter_lookback_days * 24 * 12 :]
            q3, q97 = np.quantile(trunc, [0.03, 0.97])
            clipped = s.clip(q3, q97)
            poly_detrend(clipped.values, detrend_order)
            return clipped.mad()

        jitter = by_name["low"].agg(lambda s: jitter(s))
        jitter.name = "jitter"

        q50 = meta["q50"]
        rel_jitter = jitter / q50
        rel_jitter.name = "rel_jitter"

        joint = means.join([jitter, rel_jitter])
        joint["4h_cashflow"] = joint["lvol"] * joint["low"]

        effective_limit = np.minimum(joint["lmt"], joint["lvol"])
        effective_swing = np.minimum(joint["jitter"], joint.eval("q80 - q20 - 1")).clip(
            lower=0
        )

        self.api.refresh_1h()
        self.api.refresh_5m()

        d5m = (
            joint.index.map(
                lambda x: self.api.prices_5m[x]["avgLowPrice"]
                if x in self.api.prices_5m
                else None
            ).to_series(index=joint.index)
            - q50
        ) / q50
        d5m.name = "d5m"
        d1h = (
            joint.index.map(
                lambda x: self.api.prices_1h[x]["avgLowPrice"]
                if x in self.api.prices_1h
                else None
            ).to_series(index=joint.index)
            - q50
        ) / q50
        d1h.name = "d1h"

        joint = joint.join([d1h, d5m])

        joint["profit"] = effective_swing * effective_limit
        joint["capex"] = joint["q20"] * effective_limit
        joint["roi"] = 100 * joint["profit"] / joint["capex"]

        joint.query(
            (
                f"null_ratio <= {max_null_ratio}"
                f"& jitter >= {min_jitter} "
                f"& profit >= {min_profit} "
                f"& abs(trend) <= {max_trend}"
                f"& roi >= {min_roi}"
            ),
            inplace=True,
        )

        joint["trend"] = joint["trend"].abs()

        print(joint.sort_values("profit"))

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

    def scan_drop(self):

        _, meta = self.api.load_universe(
            window_days=7,
            max_low_na_ratio=0.3,
            min_4h_vol=5000,
            min_low=50,
        )

        cur_lows = {}
        new_lows = {}

        while True:
            self.api.refresh_1h()
            self.api.refresh_5m()

            out = self.api.latest()
            new_lows.clear()

            for name in meta.index:

                high = out[name]["high"]

                base_q50 = meta.loc[name, "q50"]
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

                if drop < 0.8 or base - high > 150:
                    new_lows[name] = high, base, drop

            for name, (high, base, drop) in new_lows.items():
                if name not in cur_lows:
                    print(
                        f"{datetime.now()}: {name}: {high=:.0f}, {base=:.0f}, {drop=:.3f}"
                    )

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

    api = PriceAPI(refresh=True)
    grift = Grifter(api=api)
    grift.scan_var(min_price=5, max_price=200_000, min_4h_vol=2500)
