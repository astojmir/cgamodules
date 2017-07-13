#! /usr/bin/env python
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
# Code authors:  Aleksandar Stojmirovic and Patrick Kimes
#


import argparse
import os
import sys
import json
import cyrestclient as cyrest
from modulenet import read_modnet_info
from modulenet import load_corr_nodes
from modulenet import corr_lines_iter
from typedtable import load_edge_table
from typedtable import load_node_table


MIN_GENES = 10
node_col_names = ['ModuleID', 'Prototype', 'Probesets', 'Genes', 'NumMembers',
                  'NumProbesets', 'NumGenes']
node_col_types = [str, str, str, str, int, int, int]
edge_col_names = ['Source', 'Target', 'OverlapProbesets', 'OverlapGenes',
                  'NumOverlapProbesets', 'NumOverlapGenes']
edge_col_types = [str, str, str, str, int, int]


def _set_style(client, net_suid, style_file):
    if style_file is not None:
        with open(style_file, 'r') as fp:
            style_obj = json.load(fp)
        style_name = style_obj['title']
        if style_name not in client.get_all_style_names():
            client.save_new_style(style_obj)
        try:
            client.apply_style(style_name, net_suid)
        except:
            pass


def _autocorr_subnet(subnet_name, corr_file, psannot_file, style_file,
                     cutoff=0.0, probesets=None, primary_key='Probeset'):

    if probesets is None:
        filter_func = lambda ps: True
    else:
        filter_func = lambda ps: ps in probesets

    # Extract probesets from the annotation file
    #   First get the mapping of probesets to rows
    nodes_data, _ = load_corr_nodes(psannot_file, primary_key=primary_key)
    lbl2ix = dict((k, i) for i, k in enumerate(nodes_data))

    #   Then, load the entire rows
    with open(psannot_file, 'r') as fp:
        node_rows = load_node_table(fp, header=True)

    PG = cyrest.CytoNetworkProps(name=subnet_name)

    def _add_node(PG, lbl, nodes_data):
        i = lbl2ix[lbl]
        node_label, data = node_rows[i]
        node = PG.create_node(i, nodes_data[i])
        PG.set_node_attr(i, 'Label', node_label)
        for attr, val in zip(data._fields, data):
            PG.set_node_attr(i, attr, val)

    used_probesets = set()
    for item in corr_lines_iter(corr_file, nodes_data, cutoff):
        src_id, tgt_id, r, p = item[:4]
        if not filter_func(src_id) or not filter_func(tgt_id):
            continue
        if src_id not in used_probesets:
            _add_node(PG, src_id, nodes_data)
            used_probesets.add(src_id)
        if tgt_id not in used_probesets:
            _add_node(PG, tgt_id, nodes_data)
            used_probesets.add(tgt_id)

        _src_id = PG._get_main_node_id(src_id)
        _tgt_id = PG._get_main_node_id(tgt_id)
        obj = PG.create_edge(_src_id, _tgt_id)
        obj[cyrest.DATA]['name'] = '%s -- %s' % (src_id, tgt_id)
        obj[cyrest.DATA]['Corr'] = r
        obj[cyrest.DATA]['CorrPval'] = p
        obj[cyrest.DATA]['AbsCorr'] = abs(r)
        obj[cyrest.DATA]['CorrSign'] = 1 if r >= 0 else -1

    client = cyrest.CytoscapeClient()
    net_suid = client.save_network(PG)
    try:
        client.apply_layout('force-directed', net_suid)
    except:
        pass
    _set_style(client, net_suid, style_file)


def do_autocorr_net(args):

    style_file = args.style_file
    corr_file = args.corr_file
    psannot_file = args.psannot_file
    cutoff = float(args.abscorr_cutoff)

    dataset_prefix = corr_file.replace('.corr.txt', '')
    subnet_name = '%s (rc=%.4f)' % (dataset_prefix, cutoff)

    _autocorr_subnet(subnet_name, corr_file, psannot_file, style_file,
                     cutoff, primary_key=args.primary_key)


def do_get_style(args):

    client = cyrest.CytoscapeClient()
    obj = client.get_style(args.style_name)
    json.dump(obj, args.style_file, indent=2)


def do_modnet(args):

    modnet_file = args.modnet_file
    style_file = args.style_file

    modnet_name = os.path.basename(modnet_file).replace('.el.txt', '')
    modnet_nodes_file = modnet_file.replace('.el.txt', '.nl.txt')

    PG = cyrest.CytoNetworkProps(name=modnet_name)

    with open(modnet_nodes_file, 'r') as fp:
        node_rows = load_node_table(fp, header=False,
                                    col_names=node_col_names,
                                    col_types=node_col_types)

    for i, row in enumerate(node_rows):
        node_name, data = row
        if data[5] < MIN_GENES:
            continue
        PG.create_node(i, node_name)
        for attr, val in zip(data._fields, data):
            PG.set_node_attr(i, attr, val)

    with open(modnet_file, 'r') as fp:
        edge_rows = load_edge_table(fp, header=False,
                                    col_names=edge_col_names,
                                    col_types=edge_col_types)
    for edge, data in edge_rows:
        src_id, tgt_id = edge
        if src_id is None or tgt_id is None:
            continue
        _src_id = PG._get_main_node_id(src_id)
        _tgt_id = PG._get_main_node_id(tgt_id)
        if _src_id is None or _tgt_id is None:
            continue
        obj = PG.create_edge(_src_id, _tgt_id)
        for attr, item in zip(data._fields, data):
            obj[cyrest.DATA][attr] = item

    client = cyrest.CytoscapeClient()
    net_suid = client.save_network(PG)
    try:
        client.apply_layout('force-directed', net_suid)
    except:
        pass
    _set_style(client, net_suid, style_file)


def do_modsub(args):

    modnet_info = read_modnet_info(args.modnet_info_file)
    modnet_nodes_file = args.modnet_info_file.replace('.mod.info.txt',
                                                      '.mod.nl.txt')
    module_ids = args.module_id
    style_file = args.style_file

    corr_file = modnet_info['corr_file']
    psannot_file = modnet_info['psannot_file']
    cutoff = float(modnet_info['abscorr_cutoff'])

    with open(modnet_nodes_file, 'r') as fp:
        module_node_rows = load_node_table(fp, header=False,
                                           col_names=node_col_names,
                                           col_types=node_col_types)

    # Extract probesets from the annotation file
    #   Get the mapping of probesets to rows as used by modules
    nodes_data, _ = load_corr_nodes(psannot_file)

    # #   Then, load the entire rows
    # with open(psannot_file, 'r') as fp:
    #     node_rows = load_node_table(fp, header=True)

    # - means we get the entire network
    if len(module_ids) == 1 and module_ids[0] == '-':
        subnet_name = modnet_info['dataset_prefix'] + '-genes'
        probesets = None
    else:
        # Find probeset members of the specified modules
        subnet_name = ' & '.join(module_ids)
        probesets = set()
        for _module_id, data in module_node_rows:
            if _module_id in module_ids:
                probesets.update(data[1].split(','))
        assert len(probesets) > 0

    _autocorr_subnet(subnet_name, corr_file, psannot_file, style_file,
                     cutoff, probesets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load networks into Cytoscape')
    subparsers = parser.add_subparsers()
    parser.add_argument('-v',
                        action='store',
                        nargs='?',
                        const=1,
                        default=0,
                        type=int,
                        dest='verbose',
                        help='Verbosity level')

    # ******************************************************************

    parser1 = subparsers.add_parser('autocorr-net',
                                    help='Load a within-dataset correlation network')
    parser1.set_defaults(func=do_autocorr_net)
    parser1.add_argument('corr_file')
    parser1.add_argument('psannot_file')
    parser1.add_argument('abscorr_cutoff')
    parser1.add_argument('-s',
                         action='store',
                         default=None,
                         dest='style_file',
                         help='name of style file to apply')
    parser1.add_argument('-k', '--primary-key',
                         action='store',
                         default='Probeset',
                         dest='primary_key',
                         help='Primary key for nodes')

    # ******************************************************************

    parser2 = subparsers.add_parser('get-style',
                                    help='Save a style from Cytoscape into a file')
    parser2.set_defaults(func=do_get_style)
    parser2.add_argument('style_name', help='style name')
    parser2.add_argument('style_file', nargs='?', type=argparse.FileType('w'),
                         default=sys.stdout, help='style output file')

    # ******************************************************************

    parser3 = subparsers.add_parser('modnet',
                                    help='Load a module network')
    parser3.set_defaults(func=do_modnet)
    parser3.add_argument('modnet_file', help='module network file (el.txt)')
    parser3.add_argument('-s',
                         action='store',
                         default=None,
                         dest='style_file',
                         help='name of style file to apply')

    # ******************************************************************

    parser5 = subparsers.add_parser('modsub',
                                    help='Correlation network within modules')
    parser5.set_defaults(func=do_modsub)
    parser5.add_argument('modnet_info_file', help='.mod.info.txt file')
    parser5.add_argument('module_id', nargs='+',
                         help='module ID ("-" for entire network)')
    parser5.add_argument('-s',
                         action='store',
                         default=None,
                         dest='style_file',
                         help='name of style file to apply')

    # ******************************************************************

    args = parser.parse_args()
    args.func(args)
