import sqlite3 as sql
from time import time
from typing import Any

import pandas as pd
import requests as rq
from tqdm import tqdm

from src import PRICE_DB_FN

BASE_URL = "http://prices.runescape.wiki/api/v1/osrs"
UA = "the most sublime grift scanner v1337.69"


class PriceAPI:

    # TODO internalize some methods
    @staticmethod
    def db_conn():
        return sql.connect(PRICE_DB_FN)

    def __init__(self, do_refresh: bool = True):
        self.session = rq.Session()
        self.headers = {"User-Agent": UA}

        self.ingest_map()
        with self.db_conn() as conn:
            self.items_by_id: dict[int, dict[str, Any]] = pd.read_sql(
                "SELECT * FROM item_map", conn, index_col="id"
            ).to_dict(orient="index")

            self.items_by_name = {
                it["name"]: it | {"id": key}
                for key, it in self.items_by_id.items()
            }

        if do_refresh:
            self.refresh()

        self.time_1h = 0
        self.prices_1h = {}
        self.refresh_1h()

    def id_from_name(self, name: str) -> int:
        return self.items_by_name[name]['id']

    def name_from_id(self, id: int) -> str:
        return self.items_by_id[id]['name']

    def get(self, path: str) -> Any:
        return self.session.get(
            BASE_URL + "/" + path, headers=self.headers
        ).json()

    def ingest_map(self):
        """
        Refreshes the item map from the API.
        """
        raw_map = self.get("mapping")
        with self.db_conn() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE
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
        pre_filtered = {k: v for k, v in self.items_by_id.items()}

        for key in tqdm(pre_filtered):
            self._ingest_price(key)

    def refresh_1h(self):
        if time() - self.time_1h > 60:
            self.prices_1h = self.get("1h")

