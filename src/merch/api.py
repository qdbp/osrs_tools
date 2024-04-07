import sqlite3 as sql
from datetime import datetime, timedelta
from functools import partial
from itertools import product
from time import time
from typing import Any

import pandas as pd
import requests as rq
from pandas import DataFrame
from tqdm import tqdm

from src import PRICE_DB_FN
from src.merch.recipe import Item

BASE_URL = "http://prices.runescape.wiki/api/v1/osrs"
UA = "the most sublime grift scanner v1337.69"

# samples per hour
SECS_4H = 4 * 60 * 60


class PriceAPI:

    # TODO internalize some methods
    @staticmethod
    def db_conn():
        return sql.connect(PRICE_DB_FN)

    def __init__(self, refresh: bool = True):
        self.session = rq.Session()
        self.headers = {"User-Agent": UA}

        self.ingest_map()

        with self.db_conn() as conn:
            raw_items: dict[int, dict[str, Any]] = pd.read_sql(
                "SELECT * FROM item_map", conn, index_col="id"
            ).to_dict(orient="index")

        self.by_id: dict[int, Item] = {
            k: Item(id=k, name=v["name"], limit=v["lmt"], value=v["value"]) for k, v in raw_items.items()
        }

        self.by_name: dict[str, Item] = {item.name: item for item in self.by_id.values()}

        if refresh:
            self.refresh()

        self.time_1h = 0
        self.prices_1h: dict[str, dict[str, Any]] = {}
        self.refresh_1h()

        self.time_5m = 0
        self.prices_5m: dict[str, dict[str, Any]] = {}
        self.refresh_5m()

    def id_from_name(self, name: str) -> int:
        return self.by_name[name].id

    def name_from_id(self, id: int) -> str:
        return self.by_id[id].name

    def get(self, path: str) -> Any:
        return self.session.get(BASE_URL + "/" + path, headers=self.headers).json()

    def ingest_map(self):
        """
        Refreshes the item map from the API.
        """
        raw_map = self.get("mapping")
        with self.db_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE
                INTO item_map (id, name, examine, lmt, value, members) 
                VALUES (:id, :name, :examine, :limit, :value, :members)
                """,
                ({"limit": None, "value": None} | it for it in raw_map),
            )

    def _ingest_price(self, item_id: int):
        """
        Reads a single price time series for a given item id.
        """
        raw_prices = self.get(f"timeseries?id={item_id}&timestep=5m")["data"]
        with self.db_conn() as conn:
            conn.executemany(
                f"""
                INSERT OR IGNORE INTO prices (id, ts, high, hvol, low, lvol)
                VALUES (
                    {item_id}, :timestamp, :avgHighPrice, :highPriceVolume, 
                    :avgLowPrice, :lowPriceVolume
                )
                """,
                raw_prices,
            )

    def refresh(self):
        """
        Ingests prices for all known items into the database.
        """
        # noinspection PyChainedComparisons
        pre_filtered = {k: v for k, v in self.by_id.items()}

        for key, item in (
            progress := tqdm(sorted(pre_filtered.items(), key=lambda x: -(x[1].limit or 1)))
        ):
            self._ingest_price(key)
            progress.desc = f"{item.name: >30s}: {item.limit}: "

    def refresh_1h(self):
        if time() - self.time_1h > 60:
            self.prices_1h = {self.name_from_id(int(k)): v for k, v in self.get("1h")["data"].items()}

    def refresh_5m(self):
        if time() - self.time_1h > 10:
            self.prices_5m = {self.name_from_id(int(k)): v for k, v in self.get("5m")["data"].items()}

    def latest(self, name: str = None):

        query = "" if name is None else f"?id={self.id_from_name(name)}"

        out = self.get("latest" + query)["data"]
        out = {
            self.name_from_id(int(k)): v
            for k, v in out.items()
            if int(k) in self.by_id
            and v["highTime"] is not None
            and v["lowTime"] is not None
            and "low" in v
            and "high" in v
        }
        for v in out.values():
            v["highTime"] = datetime.fromtimestamp(v["highTime"])
            v["lowTime"] = datetime.fromtimestamp(v["lowTime"])

        return out if name is None else out[name]

    def load_universe(
        self,
        window_days: float = 14,
        quantile_days: float = 7,
        min_4h_vol: float = 1000,
        min_lmt: float = 1,
        min_low: float = 5,
        max_low: float = 10_000,
        max_null_ratio: float = 0.5,
        min_samples: int = 20,
    ) -> tuple[DataFrame, DataFrame]:

        cutoff = int(time()) - window_days * 86400
        price_types = ["low", "high"]

        with self.db_conn() as conn:
            prices = pd.read_sql(
                f"""
                SELECT prices.id, name, ts as time, low, high, lvol + hvol as volume, lmt
                FROM prices INNER JOIN item_map on prices.id = item_map.id
                WHERE prices.ts >= {cutoff} AND item_map.lmt >= {min_lmt}
                """,
                conn,
                parse_dates={"time": "s"},
            ).sort_values(["name", "time"])
            prices.rename(columns={"lmt": "limit"}, inplace=True)

        by_name = prices.groupby("name")
        prices = by_name.filter(
            lambda df: (
                (df["low"].isna().sum() <= max_null_ratio * len(df))
                and (min_low <= df["low"].median() <= max_low)
                and len(df) >= min_samples
            )
        )

        by_name = prices.groupby("name")
        # low always has fewer nulls
        null_ratio = by_name["low"].agg(lambda s: s.isna().sum() / len(s))
        null_ratio.name = "null_ratio"

        for which in price_types:
            prices[which] = prices[which].interpolate(method="linear")
            prices[which].ffill(inplace=True)
            prices[which].bfill(inplace=True)

        def calc_vol_4h(sdf: DataFrame):
            dt = sdf["time"].iloc[-1] - sdf["time"].iloc[0]
            tot_vol = sdf["volume"].sum()
            return tot_vol * SECS_4H / dt.total_seconds()

        vol_4h = by_name.apply(calc_vol_4h)
        vol_4h.name = "4h_vol"

        def get_quantile(q: float, which: str, cutoff: pd.Timestamp, sdf: DataFrame):
            sdf = sdf[sdf["time"] >= cutoff]
            return sdf[which].quantile(q)

        q_vals = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]
        qws = list(product(q_vals, price_types))
        q_cutoff = prices["time"].max() - timedelta(days=quantile_days)
        q_cols = [by_name[[w, "time"]].apply(partial(get_quantile, q, w, q_cutoff)) for q, w in qws]
        for (q, w), q_col in zip(qws, q_cols):
            q_col.name = f"q{q * 100:.0f}_{w}"

        meta = null_ratio.to_frame()
        meta = meta.join([by_name["limit"].agg("first"), vol_4h, *q_cols])
        meta = meta[meta["4h_vol"] >= min_4h_vol].sort_index()

        prices = prices.drop("limit", axis=1).set_index("name").sort_index()
        prices = prices.join(meta[[]], how="inner")

        return prices, meta
