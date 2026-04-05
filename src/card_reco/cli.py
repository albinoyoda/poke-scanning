from __future__ import annotations

import argparse
import sys
from pathlib import Path

from card_reco import identify_cards


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="card-reco",
        description="Identify Pokemon cards from images",
    )
    subparsers = parser.add_subparsers(dest="command")

    # identify command
    id_parser = subparsers.add_parser("identify", help="Identify cards in an image")
    id_parser.add_argument("image", type=str, help="Path to the input image")
    id_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to hash database (default: data/card_hashes.db)",
    )
    id_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of match candidates per card (default: 5)",
    )
    id_parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Max combined hash distance for a match (default: 40.0)",
    )
    id_parser.add_argument(
        "--debug",
        type=str,
        nargs="?",
        const="debug",
        default=None,
        metavar="DIR",
        help="Save debug images to DIR (default: debug/)",
    )
    id_parser.add_argument(
        "--backend",
        type=str,
        choices=["hash", "cnn"],
        default="cnn",
        help="Matching backend: hash (perceptual) or cnn (CNN+FAISS, default)",
    )

    # scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Live screen-capture scanner with GUI",
    )
    scan_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to hash database (default: data/card_hashes.db)",
    )
    scan_parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index to capture (default: 1 = primary)",
    )
    scan_parser.add_argument(
        "--region",
        type=str,
        default=None,
        metavar="X,Y,W,H",
        help="Sub-region to capture as X,Y,WIDTH,HEIGHT",
    )
    scan_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of match candidates per card (default: 5)",
    )
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Max combined hash distance for a match (default: 40.0)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "identify":
        _cmd_identify(args)
    elif args.command == "scan":
        _cmd_scan(args)


def _cmd_identify(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    debug = None
    if args.debug is not None:
        # pylint: disable-next=import-outside-toplevel
        from card_reco.debug import DebugWriter

        debug = DebugWriter(args.debug)
        print(f"Debug output: {debug.output_dir.resolve()}")

    results = identify_cards(
        image_path,
        db_path=args.db,
        top_n=args.top_n,
        threshold=args.threshold,
        debug=debug,
        backend=args.backend,
    )

    if not results:
        print("No cards detected in the image.")
        return

    for i, matches in enumerate(results, 1):
        print(f"\n--- Card {i} ---")
        if not matches:
            print("  No match found.")
            continue
        for match in matches:
            c = match.card
            print(
                f"  #{match.rank} {c.name} ({c.set_name} {c.number}) "
                f"[distance: {match.distance:.1f}]"
            )


if __name__ == "__main__":
    main()


def _cmd_scan(args: argparse.Namespace) -> None:
    # pylint: disable-next=import-outside-toplevel
    from card_reco.scanner import Scanner

    region: tuple[int, int, int, int] | None = None
    if args.region is not None:
        parts = args.region.split(",")
        if len(parts) != 4:
            print(
                "Error: --region must be X,Y,WIDTH,HEIGHT (4 comma-separated ints)",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            x, y, w, h = (int(p) for p in parts)
        except ValueError:
            print("Error: --region values must be integers", file=sys.stderr)
            sys.exit(1)
        region = (x, y, w, h)

    scanner = Scanner(
        db_path=args.db,
        monitor=args.monitor,
        region=region,
        top_n=args.top_n,
        threshold=args.threshold,
    )
    scanner.run()
