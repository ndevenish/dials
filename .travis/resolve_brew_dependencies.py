#!/usr/bin/env python
from collections import defaultdict
import json
import os
import sys
import subprocess

try:
    from packaging.requirements import Requirement
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import parse

BOLD = "\033[1m"
RED = "\033[1;31m"
MAGENTA = "\033[35m"
GREEN = "\033[32m"
NC = "\033[0m"

# A homebrew environment to specify no updating
NO_UPDATE = dict(os.environ, **{"HOMEBREW_NO_AUTO_UPDATE": "1"})


class CannotMatchVersion(Exception):
    pass


def install_or_upgrade(requirements):
    # First, interrogate homebrew to see what we have
    brew_json = subprocess.check_output(
        ["brew", "info", "--json"] + [x.name for x in requirements], env=NO_UPDATE
    )
    print("Brew output:\n"+brew_json)
    brew_data = {x["name"]: x for x in json.loads(brew_json)}

    # Can we not get a valid version without updating homebrew?
    needs_update = False

    rows = []
    actions = defaultdict(list)
    for req in requirements:
        name = req.name
        cur_ver_str = brew_data[name]["linked_keg"]
        cur_ver = parse(cur_ver_str) if cur_ver_str else None
        avail_ver = parse(brew_data[name]["versions"]["stable"])
        # Check the combinations of availability
        # action = None
        if cur_ver is None:
            if avail_ver in req.specifier:
                action = "install"
            else:
                action = "nomatch"
        else:
            if cur_ver in req.specifier:
                action = None
            elif avail_ver in req.specifier:
                action = "upgrade"
            else:
                action = "nomatch"
        rows.append(
            [
                name,
                str(cur_ver) if cur_ver is not None else "",
                str(avail_ver),
                str(req.specifier),
                action,
            ]
        )
        actions[action].append(req)

    # import pdb
    # pdb.set_trace()

    # Print a table of columns
    titles = ["Name", "Installed", "Available", "Spec", "Action"]
    column_widths = [
        max(len(str(row[i])) for row in rows + [titles]) for i in range(len(rows[0]))
    ]
    # Title
    print(
        titles[0].ljust(column_widths[0] + 1)
        + " ".join(x.rjust(i) for x, i in zip(titles[1:], column_widths[1:]))
    )
    # Each row
    COLMAP = {None: GREEN, "nomatch": RED, "upgrade": MAGENTA, "install": MAGENTA}
    colors = [COLMAP[x[-1]] for x in rows]
    for row, color in zip(rows, colors):
        name_col = row[0].ljust(column_widths[0]) + " "
        print(
            color
            + name_col
            + " ".join(
                (x if x is not None else "").rjust(i)
                for x, i in zip(row[1:], column_widths[1:])
            )
            + NC
        )

    # Do we need to update homebrew?
    if actions["nomatch"]:
        raise CannotMatchVersion(
            "Cannot match versions for " + ", ".join(str(x) for x in actions["nomatch"])
        )

    return actions["install"], actions["upgrade"]


requirements = [Requirement(x) for x in sys.argv[1:]]

try:
    print("Homebrew requirement compatibility:")
    install, upgrade = install_or_upgrade(requirements)
except CannotMatchVersion:
    print(
        RED
        + "Error:"
        + NC
        + " Cannot resolve with current brew version. Assuming is upgrade-related."
    )
    print("Running 'brew update':")
    subprocess.check_call(["brew", "update"])
    try:
        print("Trying install with updated homebrew:")
        install, upgrade = install_or_upgrade(requirements)
    except CannotMatchVersion as e:
        print(RED + "Error: " + str(e))
        sys.exit(1)


def _intercept(args, env=None):
    print("Running: " + " ".join(args))

if install:
    print(GREEN + BOLD + "Installing: " + NC + " ".join(x.name for x in install))
    cmd = ["brew", "install"] + [x.name for x in install]
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, env=NO_UPDATE)

if upgrade:
    print(GREEN + BOLD + "Upgrading: " + NC + " ".join(x.name for x in upgrade))
    cmd = ["brew", "upgrade"] + [x.name for x in upgrade]
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, env=NO_UPDATE)
