# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>
# -----------------------------------------------------------------------

#
# Code author:  Aleksandar Stojmirovic
#

COLUMN_TYPES = {'Integer': int,
                'Long': int,
                'Double': float,
                'Boolean': bool,
                }


class TableRow(list):
    """
    A fix around namedtuple's limitation to 255 fields
    """

    __slots__ = ['_fields']

    def __init__(self, attrs, values):
        assert len(attrs) == len(values)
        list.__init__(self, values)
        self._fields = attrs


def _process_data_row(line, num_cols):
    _fields = line.rstrip('\n').split('\t')
    fields = [s if s else None for s in _fields]
    if len(fields) < num_cols:
        fields += [None] * (num_cols - len(fields))
    assert len(fields) == num_cols
    return fields


def load_typed_table(fp, header=True, col_names=None, col_types=None):
    """
    Load table from a tab-delimited file, performing type conversion if
    required.

    If header is True, the first line is assumed to be a header.
    If header is True and the second line starts with '#', it is used to
    determine the type for each column (in Java style).
    """

    if col_names is not None:
        header = False

    num_cols = None
    _rows = []
    if header:
        col_names = next(fp).rstrip('\n').split('\t')
        num_cols = len(col_names)
        line = next(fp)
        if line[0] == '#':
            fields = line[1:].rstrip('\n').split('\t')
            assert len(fields) <= num_cols
            col_types = [COLUMN_TYPES.get(s, str) for s in fields]
        else:
            _rows.append(_process_data_row(line, num_cols))
    else:
        line = next(fp)
        fields = line.rstrip('\n').split('\t')
        num_cols = len(fields)
        _rows.append(_process_data_row(line, num_cols))

    for line in fp:
        if line[0] == '#':
            continue
        _rows.append(_process_data_row(line, num_cols))

    if col_names is None:
        col_names = ['Column%2.2d' % j for j in range(1, num_cols+1)]

    if col_types is not None:
        rows = [[ct(s) if s is not None else None
                 for ct,s in zip(col_types, fields)] for fields in _rows]
    else:
        rows = _rows

    return col_names, rows


def load_edge_table(fp, src_col=0, tgt_col=1, header=True, col_names=None,
                    col_types=None):
    """
    Load edges and their attributes from a tab-delimited file.

    If header is True, the first line is assumed to be a header.
    If header is True and the second line starts with '#', it is used to
    determine the type for each column (in Java style).
    """

    _attrs, _rows = load_typed_table(fp, header, col_names, col_types)
    special_cols = (src_col, tgt_col)
    attrs = tuple(s for i, s in enumerate(_attrs) if i not in special_cols)

    rows = []
    for fields in _rows:
        edge = tuple(fields[j] for j in special_cols)
        values = tuple(s for j, s in enumerate(fields)
                       if j not in special_cols)
        data = TableRow(attrs, values)
        rows.append((edge, data))
    return rows


def load_node_table(fp, id_col=0, header=True, col_names=None, col_types=None):
    """
    Load nodes and their attributes from a tab-delimited file.

    If header is True, the first line is assumed to be a header.
    If header is True and the second line starts with '#', it is used to
    determine the type for each column (in Java style).
    """

    _attrs, _rows = load_typed_table(fp, header, col_names, col_types)
    attrs = tuple(s for i, s in enumerate(_attrs) if i != id_col)

    rows = []
    for fields in _rows:
        node = fields[id_col]
        values = tuple(s for j, s in enumerate(fields) if j != id_col)
        data = TableRow(attrs, values)
        rows.append((node, data))
    return rows


def load_list(fp, attr_name='ListAttribute'):
    """
    Load a list of identifiers from a file as a node table.

    Output is in the same format as for load_node_table(). Each row will have
    only one data column set to True, with a name given by attr_name.
    """

    _attrs, _rows = load_typed_table(fp, header=False)
    assert len(_attrs) == 1

    rows = []
    for fields in _rows:
        node = fields[0]
        data = TableRow(tuple([attr_name]), tuple([True]))
        rows.append((node, data))
    return rows
