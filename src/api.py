import sqlite3 as sql
from datetime import datetime
from time import time
from typing import Any

import pandas as pd
import requests as rq
import numpy as np
from pandas import DataFrame, Series
from scipy.stats import linregress, siegelslopes
from tqdm import tqdm

from src import PRICE_DB_FN
from src.recipe import Item

BASE_URL = "http://prices.runescape.wiki/api/v1/osrs"
UA = "the most sublime grift scanner v1337.69"

# samples per hour
SPH = 12


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
            k: Item(id=k, name=v["name"], limit=v["lmt"], value=v["value"])
            for k, v in raw_items.items()
        }

        self.by_name: dict[str, Item] = {
            item.name: item for item in self.by_id.values()
        }

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
            progress := tqdm(
                sorted(pre_filtered.items(), key=lambda x: -(x[1].limit or 1))
            )
        ) :
            self._ingest_price(key)
            progress.desc = f"{item.name: >30s}: {item.limit}: "

    def refresh_1h(self):
        if time() - self.time_1h > 60:
            self.prices_1h = {
                self.name_from_id(int(k)): v for k, v in self.get("1h")["data"].items()
            }

    def refresh_5m(self):
        if time() - self.time_1h > 10:
            self.prices_5m = {
                self.name_from_id(int(k)): v for k, v in self.get("5m")["data"].items()
            }

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
        min_4h_vol: float = 1000,
        min_lmt: float = 1,
        min_low: float = 5,
        max_low: float = 10_000,
        max_low_na_ratio: float = 0.5,
        max_trend: float = float("inf"),
    ) -> tuple[DataFrame, DataFrame]:

        cutoff = int(time()) - window_days * 86400

        with self.db_conn() as conn:
            prices = pd.read_sql(
                f"""
                SELECT prices.id, name, ts, high, low, hvol, lvol, lmt
                FROM prices INNER JOIN item_map on prices.id = item_map.id
                WHERE prices.ts >= {cutoff} AND item_map.lmt >= {min_lmt}
                """,
                conn,
                parse_dates={"ts": "s"},
            )

        prices.sort_index()

        contiguous_ix = pd.date_range(
            prices["ts"].min(), prices["ts"].max(), freq="5min"
        )
        prices.set_index(["name", "ts"], inplace=True)
        prices = prices.reindex(contiguous_ix, level=1)
        prices.sort_index(inplace=True)

        # fillna before filter but after reindex to include zeros properly
        prices.fillna({"hvol": 0, "lvol": 0}, inplace=True)

        by_name = prices.groupby("name")
        prices = by_name.filter(
            lambda df: (
                (df["lvol"].median() * SPH * 4 >= min_4h_vol)
                and (df["low"].isna().sum() <= max_low_na_ratio * len(df))
                and (min_low <= df["low"].median() <= max_low)
            )
        )

        by_name = prices.groupby("name")
        null_ratio = by_name["low"].agg(lambda s: s.isna().sum() / len(s))
        null_ratio.name = "null_ratio"

        prices.bfill(inplace=True)

        vol_4h = by_name["lvol"].agg("mean") * SPH * 4
        vol_4h.name = "4h_vol"

        def get_trend(s: Series):
            q3, q97 = np.quantile(s, [0.03, 0.97])
            s = s.clip(q3, q97)
            slope, *_ = linregress(
                np.arange(0, len(s)) / 288, s.values / (25 + s.median())
            )
            return slope

        trend = by_name["low"].agg(lambda s: get_trend(s))
        trend.name = "trend"

        q20 = by_name["low"].agg(lambda s: s.quantile(0.20))
        q20.name = "q20"
        q50 = by_name["low"].agg(lambda s: s.quantile(0.50))
        q50.name = "q50"
        q80 = by_name["low"].agg(lambda s: s.quantile(0.80))
        q80.name = "q80"

        meta = null_ratio.to_frame()
        meta = meta.join([by_name["lmt"].agg("first"), vol_4h, trend, q20, q50, q80])

        meta = meta[meta["trend"] <= max_trend]
        prices = prices.join(meta[[]], how="inner")

        return prices, meta
