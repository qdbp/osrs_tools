import sys
from datetime import datetime
from time import sleep

import requests as rq

BASE_URL = "http://prices.runescape.wiki/api/v1/osrs"
UA = "the most sublime grift scanner v1337.69"


def main():

    id = sys.argv[1]

    with rq.Session() as sess:

        last_high = None
        last_low = None

        while True:
            out = sess.get(
                BASE_URL + f"/latest?id={id}", headers={"User-Agent": UA}
            ).json()["data"][id]

            ht = out['highTime'] = datetime.fromtimestamp(out['highTime'])
            lt = out['lowTime'] = datetime.fromtimestamp(out['lowTime'])

            if ht != last_high:
                last_high = ht
                print('new high:', out['high'], ht)
            if lt != last_low:
                last_low = lt
                print('new low:', out['low'], lt)
            sleep(3)


if __name__ == "__main__":
    main()
