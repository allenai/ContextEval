r"""Utilties for reading and writing files.

Expected format for TSV file is that each line has one rows, with each
element separated by \t. The number of element should be the same as
expected_num_columns.

Expected format for rows in memory is a list where each element is:
(element_1, element_2, ...), or [element_1, element_2, ...]
The number of element should be the same as expected_num_columns.

This module also handles the case of writing simple newline-separated txt files.
"""

import csv
import sys

# from tensorflow.io import gfile
csv.field_size_limit(sys.maxsize)


def read_tsv(filepath, delimiter="\t", max_splits=-1):
    """Read file to list of rows."""
    rows = []
    with open(filepath, "r") as tsv_file:
        for line in tsv_file:
            line = line.rstrip()
            cols = line.split(delimiter, max_splits)
            rows.append(cols)
    print("Loaded %s rows from %s." % (len(rows), filepath))
    return rows


def write_tsv(rows, filepath, delimiter="\t"):
    """Write rows to tsv file."""
    with open(filepath, "w") as tsv_file:
        for row in rows:
            line = "%s\n" % delimiter.join([str(elem) for elem in row])
            tsv_file.write(line)
    print("Wrote %s rows to %s." % (len(rows), filepath))


def read_csv(filepath):
    with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rows = list(reader)
    print("Loaded %s rows from %s." % (len(rows), filepath))
    return rows


def write_csv(rows, filepath):
    print("Writing %d lines to %s" % (len(rows), filepath))
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)


def write_txt(rows, filepath):
    """Write newline separated text file."""
    with open(filepath, "w") as tsv_file:
        for row in rows:
            line = "%s\n" % row
            tsv_file.write(line)
    print("Wrote %s rows to %s." % (len(rows), filepath))


def read_txt(filepath):
    """Read newline separated text file."""
    rows = []
    with open(filepath, "r") as tsv_file:
        for line in tsv_file:
            line = line.rstrip()
            rows.append(line)
    print("Loaded %s rows from %s." % (len(rows), filepath))
    return rows
