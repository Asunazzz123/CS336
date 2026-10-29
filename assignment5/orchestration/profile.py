"""Preview or explicitly execute an experiment profile; preview is the default."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex

from utils.paths import PROJECT_DIR


def profile_command(path, overrides=()):
    profile = json.loads(Path(path).read_text())
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a JSON object")
    values = profile.get("environment")
    if not isinstance(values, dict):
        raise ValueError("Profile requires an environment object")
    values = dict(values)
    for override in overrides:
        name, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides must use NAME=value")
        values[name] = value
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not isinstance(value, str):
            raise ValueError("Environment keys must be uppercase identifiers and values must be strings")
    command = ["bash", str(PROJECT_DIR / "orchestration/sft_rl_pipeline.sh")]
    return command, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=value")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually launch the pipeline")
    mode.add_argument("--dry-run", dest="execute", action="store_false")
    parser.set_defaults(execute=False)
    args = parser.parse_args(argv)
    try:
        command, values = profile_command(args.profile, args.overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps({"execute": False, "environment": values, "command": command,
                          "display": shlex.join(["env", *(f"{k}={v}" for k, v in values.items()), *command])},
                         ensure_ascii=False, indent=2))
        return
    environment = os.environ.copy()
    environment.update(values)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
