import sys

from configparser import ConfigParser
import semver

SETUP_FILE = "setup.cfg"
NONE = "none"
VALID_BUMPS = ["major", "minor", "patch", "prerelease", NONE]


# -----------------------------------
def _get_bump_type(args):
    valid_bumps = (
        "The bump type must be one of the following strings:\n"
        f"\t-'{VALID_BUMPS[0]}'\n"
        f"\t-'{VALID_BUMPS[1]}'\n"
        f"\t-'{VALID_BUMPS[2]}'\n"
        f"\t-'{VALID_BUMPS[3]}'\n"
        f"\t-'{VALID_BUMPS[4]}'\n"
    )

    if len(args) < 2:
        raise ValueError(f"ERROR: Running bump_version.py without passing the bump type. {valid_bumps}")

    bump_type = sys.argv[1]

    if bump_type not in VALID_BUMPS:
        raise ValueError(f"{valid_bumps}")

    return bump_type

# -----------------------------------
def _bump_version(version:str, bump_type:str):
    new_version = version
    if bump_type == "major":
        new_version = semver.bump_major(version)
    elif bump_type == "minor":
        new_version = semver.bump_minor(version)
    elif bump_type == "patch":
        new_version = semver.bump_patch(version)
    elif bump_type == "prerelease":
        new_version = semver.bump_prerelease(version)

    return new_version


# -----------------------------------
def _update_version(bump_type:str):
    config = ConfigParser()
    config.read(SETUP_FILE)

    try:
        version = config["metadata"]["version"]
    except:
        raise ValueError(
            f"ERROR: Unable to fetch package version from {SETUP_FILE} file. "
            +"Make sure the version is under the 'version' key."
        )

    new_version = _bump_version(version, bump_type)

    config["metadata"]["version"] = new_version
    with open(SETUP_FILE, 'w') as configfile:
        config.write(configfile)

    return new_version


# -----------------------------------
bump_type = _get_bump_type(sys.argv)
final_version = _update_version(bump_type)
print(final_version)