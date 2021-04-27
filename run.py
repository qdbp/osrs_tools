#!/usr/bin/python

from argparse import ArgumentParser

from src.api import PriceAPI
from src.grifter import Grifter


def main():

    parser = ArgumentParser()
    parser.add_argument("cmd", type=str, help="command to run")
    parser.add_argument("--refresh", "-r", action="store_true")

    parser.add_argument(
        "--min-profit", "-mp", type=int, help="minimum profit", default=500
    )
    parser.add_argument("--min-price", type=int, help="minimum profit", default=5)
    parser.add_argument("--max-price", type=int, help="max price", default=15_000)
    parser.add_argument("--min-vol", type=int, help="minimum profit", default=2500)
    parser.add_argument("--max-null", type=float, help="max null ratio", default=0.3)
    parser.add_argument("--items", "-i", type=str, nargs='+')

    args = parser.parse_args()

    api = PriceAPI(refresh=args.refresh)
    grifter = Grifter(api=api)

    print("Grifter is here.")

    if args.cmd == "var":
        grifter.scan_var(
            min_price=args.min_price,
            max_price=args.max_price,
            min_4h_vol=args.min_vol,
            max_null_ratio=args.max_null,
        )
    elif args.cmd == "ha":
        grifter.scan_ha(args.min_profit)
    elif args.cmd == "drop":
        print("Scanning for price drops...")
        grifter.scan_drop()
    elif args.cmd == 'feed':
        grifter.live_feed([it.capitalize() for it in args.items])
    else:
        raise NotImplementedError(f"command {args.cmd} not understood")


if __name__ == "__main__":
    main()
