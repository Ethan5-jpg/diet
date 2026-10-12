#!/usr/bin/env python3
"""Read saved, fingerprint-verified aggregate result screens."""
import argparse
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--screen", type=int, choices=[1, 2, 3])
    a = p.parse_args()
    from crossscale.reporting import saved_screens
    try:
        screens = saved_screens(a.run_dir)
        for i, screen in enumerate(screens, 1):
            if a.screen is None or a.screen == i:
                print(screen)
                print()
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print("DISPLAY STOPPED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
