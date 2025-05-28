"""Utilities for reading and writing jsonl files."""

import json


def read(filepath, limit=None, verbose=False):
    """Read jsonl file to a List of Dicts."""
    data = []
    with open(filepath, "r") as jsonl_file:
        for idx, line in enumerate(jsonl_file):
            if limit is not None and idx >= limit:
                break
            if verbose and idx % 100 == 0:
                # Print the index every 100 lines.
                print("Processing line %s." % idx)
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("Failed to parse line: `%s`" % line)
                raise e
    print("Loaded %s lines from %s." % (len(data), filepath))
    return data


def write(filepath, rows, append=False, verbose=True):
    """Write a List of Dicts to jsonl file."""
    open_mode = "a" if append else "w"
    with open(filepath, open_mode) as jsonl_file:
        for row in rows:
            line = "%s\n" % json.dumps(row)
            jsonl_file.write(line)
    if verbose:
        print("Wrote %s lines to %s." % (len(rows), filepath))
