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
from operator import itemgetter
import copy
import io
import numpy as np
from scipy.stats import hypergeom


class GenericNode(object):
    def __init__(self, node_id):
        self.node_id = node_id


class GenericEdge(object):
    def __init__(self, edge_key):
        nodes = tuple(edge_key)
        if len(nodes) == 2:
            src_id, tgt_id = nodes
        else:
            src_id = tgt_id = nodes
        self.src_id = src_id
        self.tgt_id = tgt_id


class MapGraph(dict):

    def __init__(self, node_class=GenericNode, edge_class=GenericEdge,
                 undirected=True):

        dict.__init__(self)
        self.node_class = node_class
        self.edge_class = edge_class
        self.nodes = {}
        self.edges = {}
        self.edge_keyfunc = frozenset if undirected else tuple
        self.undirected = undirected

    def insert_node(self, node_id):

        if node_id not in self.nodes:
            node = self.node_class(node_id)
            self.nodes[node_id] = node
            self[node_id] = {}
        else:
            node = self.nodes[node_id]
        return node

    def insert_edge(self, src_id, tgt_id):

        src = self.insert_node(src_id)
        tgt = self.insert_node(tgt_id)
        key = self.edge_keyfunc((src_id, tgt_id))

        if key not in self.edges:
            edge = self.edge_class(key)
            self.edges[key] = edge
            self[src_id][tgt_id] = edge
            if self.undirected:
                self[tgt_id][src_id] = edge
        else:
            edge = self.edges[key]
        return edge

    def delete_node(self, node_id):
        # This works only for undirected
        assert self.undirected

        self.pop(node_id)
        self.nodes.pop(node_id)

    def delete_edge(self, src_id, tgt_id):

        key = self.edge_keyfunc((src_id, tgt_id))
        if key in self.edges:
            self[src_id].pop(tgt_id)
            if self.undirected:
                self[tgt_id].pop(src_id)
            self.edges.pop(key)

    def write(self, out_prefix):

        node2index = dict((nid, i + 1)
                          for i, nid in enumerate(sorted(self.nodes)))
        node_file = out_prefix + '.nl.txt'
        edge_file = out_prefix + '.el.txt'
        with open(node_file, 'wb') as fp:
            for nid, i in sorted(node2index.items(), key=itemgetter(1)):
                fp.write('%d\t%s\n' % (i, nid))
        with open(edge_file, 'wb') as fp:
            for key in self.edges:
                src_id, tgt_id = tuple(key)
                fp.write('%d\t%d\n' % (node2index[src_id], node2index[tgt_id]))

    def update_with_copy(self, other, selected_nodes=None):
        """
        Copy the subgraph of other induced by selected_nodes into self.
        """

        assert self.node_class == other.node_class
        assert self.edge_class == other.edge_class
        assert self.undirected == other.undirected

        if selected_nodes is None:
            selected_nodes = set(other.nodes)
        else:
            selected_nodes = set(selected_nodes)

        # Copy all nodes
        for node_id in selected_nodes:
            assert node_id not in self.nodes
            node = copy.deepcopy(other.nodes[node_id])
            self.nodes[node_id] = node
            self[node_id] = {}

        # Copy all edges
        for src_id in selected_nodes:
            neighbors = (y for y in other[src_id] if y in selected_nodes)
            for tgt_id in neighbors:
                key = self.edge_keyfunc((src_id, tgt_id))
                edge = copy.deepcopy(other.edges[key])
                # Note that we do not have any special case for undirected here
                self.edges[key] = edge
                self[src_id][tgt_id] = edge

    def get_neighbors(self, src_id):
        """
        Get neighbors of a node
        """
        return set(tgt_id for tgt_id in self[src_id])

    def get_nbhd_ball(self, seed_id, k=1):
        """
        Get a k-ball around the seed_id node
        """
        nbhd = set([seed_id])
        seed = set([seed_id])
        for i in range(k):
            new_seed = set()
            for node in seed:
                new_seed |= self.get_neighbors(node)
            seed = new_seed
            nbhd |= new_seed
        return nbhd

    def get_nbhd_sphere(self, seed_id, k=1):
        """
        Get a k-sphere around the seed_id node
        """
        nbhd = set([seed_id])
        seed = set([seed_id])
        for i in range(k):
            nbhd |= seed
            new_seed = set()
            for node in seed:
                new_seed |= self.get_neighbors(node)
            seed = new_seed - nbhd
        return seed

    def subgraph(self, nodes):
        """
        Extract a subgraph induced by nodes
        """
        H = self.__class__(node_class=self.node_class,
                           edge_class=self.edge_class,
                           undirected=self.undirected)
        H.update_with_copy(self, nodes)
        return H

    def filtered_graph(self, excluded_nodes):
        """
        Extract a subgraph induced by the complement
        """
        excluded_nodes = set(excluded_nodes)
        included_nodes = [y for y in self.nodes if y not in excluded_nodes]
        return self.subgraph(included_nodes)

    def connected_components(self):
        """
        Compute connected components
        """
        assert self.undirected

        nodes = set(self.nodes)
        components = list()
        while nodes:
            w = nodes.pop()
            _visited = set([w])
            _unvisited = set(self[w])

            while _unvisited:
                u = _unvisited.pop()
                _visited.add(u)
                nodes.discard(u)
                if u in self:
                    _unvisited.update(v for v in self[u] if v not in _visited)
            components.append(sorted(_visited))
        components.sort(key=lambda _lst: (len(_lst), ''.join(_lst)))
        return components


class CorrelationLink(object):
    def __init__(self, edge_key):
        nodes = tuple(edge_key)
        if len(nodes) == 2:
            src_id, tgt_id = nodes
        else:
            src_id = tgt_id = nodes
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.corr = 0.0
        self.abscorr = 0.0


def load_corr_nodes(nodes_file, primary_key='Probeset',
                    other_key='GeneSymbol'):
    """
    Load node names from psannot file
    """
    nodes_data = []
    nodes_map = {}
    with open(nodes_file, 'r', encoding='utf-8') as fp:
        header = next(fp).strip().split('\t')
        try:
            probeset_col = header.index(primary_key)
            probeset_func = lambda i, _fields: _fields[probeset_col]
        except ValueError:
            probeset_col = 0
            probeset_func = lambda i, _fields: 'R%5.5d' % i
        try:
            gene_col = header.index(other_key)
        except ValueError:
            gene_col = 0

        for i, line in enumerate(fp):
            fields = line.rstrip('\n').split('\t')
            ps = probeset_func(i + 1, fields)
            nodes_data.append(ps)
            nodes_map[ps] = fields[gene_col]

    return nodes_data, nodes_map


def corr_lines_iter(edges_file, nodes_data, abscor_cutoff):
    """
    Iterator for the edges lines from correlation file
    """
    with open(edges_file, 'r', encoding='utf-8') as fp:
        for i in range(4):
            next(fp)
        for line in fp:
            fields = line.strip().split('\t')
            src, tgt = tuple(map(int, fields[:2]))
            r, p = tuple(map(float, fields[2:4]))
            if abs(r) < abscor_cutoff:
                continue
            src_id = nodes_data[src]
            tgt_id = nodes_data[tgt]
            yield (src_id, tgt_id, r, p)


def load_corr_edges(edges_file, nodes_file, abscor_cutoff):
    """
    Load coexpression edges from an edge list and
    create a network suitable for use by modnet
    """

    nodes_data, nodes_map = load_corr_nodes(nodes_file)
    G = MapGraph(GenericNode, CorrelationLink)

    for item in corr_lines_iter(edges_file, nodes_data, abscor_cutoff):
        src_id, tgt_id, rho = item[:3]
        src_node = G.insert_node(src_id)
        tgt_node = G.insert_node(tgt_id)
        if src_id == tgt_id:
            continue
        edge = G.insert_edge(src_id, tgt_id)
        edge.corr = rho
        edge.abscorr = abs(rho)

    return G, nodes_map


def get_periphery(U):
    """
    Periphery nodes
    """

    def _core_deg(x):
        return sum(1 for y in U[x] if y not in periphery)

    periphery = set()
    candidates = set(x for x in U.nodes if _core_deg(x) <= 1)
    while candidates:
        periphery |= candidates
        adjacent = set(y for x in candidates for y in U[x]
                       if y not in periphery and _core_deg(y) <= 1)
        candidates = adjacent
    return periphery


class ModuleNode(object):
    """
    Coexpression module
    """

    header = ('ModuleID', 'Prototype', 'MemberProbesets', 'MemberGenes',
              'NumMembers', 'NumMemberProbesets', 'NumMemberGenes')

    def __init__(self, node_id):

        self.node_id = node_id
        self.members = set()
        self.prototype = None

    def write_line(self, fp, probesets2genes):

        pp = '-'
        if self.prototype is not None:
            pp = self.prototype
        _members = set(p.node_id for p in self.members)
        mp = sorted(_members)
        mg = sorted(g for g in set(probesets2genes[p] for p in _members)
                    if g != '')

        items = [self.node_id, pp, ','.join(mp), ','.join(mg),
                 str(len(self.members)), str(len(mp)), str(len(mg))]
        fp.write('\t'.join(items))
        fp.write('\n')


class ModuleLink(object):
    """
    Object to hold edge information
    """

    header = ('Source', 'Target', 'OverlapProbesets', 'OverlapGenes',
              'NumOverlapProbesets', 'NumOverlapGenes')

    def __init__(self, edge_key):

        src_id, tgt_id = tuple(edge_key)
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.overlap = set()
        self.num_overlaps = 0

    def write_line(self, fp, probesets2genes):

        _overlap = set(p.node_id for p in self.overlap)
        op = sorted(_overlap)
        og = sorted(set(probesets2genes[p] for p in _overlap))

        items = [self.src_id, self.tgt_id, ','.join(op), ','.join(og),
                 str(len(op)), str(len(og))]
        fp.write('\t'.join(items))
        fp.write('\n')


class _ModuleInserter(object):

    def __init__(self, U, M, dataset_prefix=None):
        self.num_modules = 0
        self.M = M
        self.U = U
        self.prefix = ''
        if dataset_prefix is not None:
            self.prefix = '%s:' % dataset_prefix

    def __call__(self, member_ids, prototype):
        self.num_modules += 1
        mod_id = '%sM%3.3d' % (self.prefix, self.num_modules)
        node = self.M.insert_node(mod_id)
        member_set = set(self.U.nodes[k] for k in member_ids)
        node.members |= member_set
        node.prototype = prototype
        return mod_id


def _expand_node(U, node):
    return [p for p in U.nodes[node].members]


def modnet(U, alpha=0.01, min_mod_size=6, tau=0.75, verbose=0,
           consider_core_only=True, dataset_prefix=None, steps=1):
    """
    Construct module network

    Take an undirected MapGraph instance and return a lower dimensional module
    network representation as the output.

    :param U: an undirected graph (without self-pointing edges)
    :param alpha: a numeric value specifying the hypergeometric p-value cutoff,
    :param min_mod_size: a numeric value specifying the minimum degree for a
           node to be considered as a candidate prototype for defining a module
    :param tau: a numeric value between (0, 1) corresponding to the allowed
           proportional decrease in `min_mod_size` for connected components
           with more than `min_mod_size` nodes, but for which the maximal
           degree is less than `min_mod_size`
    :returns M: an undirected  MapGraph instance with modules as nodes. Two
           modules are connected by an edge if they share at least one
           overlapping node.

    Details:

    The algorithm seeks to identify a low-rank representation of a larger gene
    coexpression network, i.e. a representation with relatively few nodes and
    connections. Given a graph, the approach greedily identifies modules based
    on neighborhoods of the high-degree nodes. We refer to these high degree
    nodes as `prototypes` which serve as the center of each module.

    To determine the set of prototypes and modules, we apply a hypergeometric
    test to determine whether candidates significantly overlap with existing
    modules. If not, we construct a new module. After identifying all core
    modules, peripheral nodes which do not fall into k-step neighborhoods of
    any module are assigned to modules of the nearest prototypes.
    """

    M = MapGraph(node_class=ModuleNode, edge_class=ModuleLink, undirected=True)

    _insert_module = _ModuleInserter(U, M, dataset_prefix)

    # Determine periphery
    periphery = set()
    if consider_core_only:
        periphery = get_periphery(U)

    if verbose >= 1:
        print("COMPUTED PERIPHERY")

    # Extract core graph
    C = U.filtered_graph(periphery)
    Cdegrees = dict((x, len(C[x])) for x in C)

    if verbose >= 1:
        print("EXTRACTED CORE")

    # Determine connected components
    # These should be sorted in increasing order
    cmpns = U.connected_components()
    cmpns.sort(key=lambda x: len(x), reverse=True)

    if verbose >= 1:
        print("CONNECTED COMPONENTS = %d" % len(cmpns))

    for i, cmpn in enumerate(cmpns):

        if verbose >= 1:
            print("  COMPONENT #%d (%d)" % (i + 1, len(cmpn)))

        if len(cmpn) < min_mod_size:
            # ** Small connected components are not broken into modules
            _insert_module(cmpn, None)
            if verbose >= 1:
                print("    NOT BROKEN")

        else:
            # ** Break-up large connected components
            if verbose >= 1:
                print("    BREAKING")

            # Use lower cutoff if maximum degree with the component is too low
            sg_mod = min_mod_size
            core_nodes = [x for x in cmpn if x in C]
            if core_nodes:
                max_degree = max(Cdegrees[x] for x in cmpn if x in C)
            else:
                max_degree = 0
            if max_degree < min_mod_size:
                sg_mod *= tau
            if verbose >= 2:
                print("    max_degree = %d" % max_degree)

            # Compute candidates for module centers
            # Candidates are sorted by decreasing degree but ties (nodes with
            # the same degree) are not handled in any particular way (sort is
            # alphabetical)
            candidates = [(x, Cdegrees[x], C.get_nbhd_ball(x, k=steps))
                          for x in core_nodes if Cdegrees[x] >= sg_mod]
            candidates.sort(key=lambda item: (len(item[2]), item[1], item[0]),
                            reverse=True)

            if verbose >= 1:
                print("    COMPUTED %d CANDIDATES" % len(candidates))

            # Insert entire module if no candidates can be found
            if not candidates:
                _insert_module(cmpn, None)
                if verbose >= 1:
                    print("    NOT BROKEN")
                continue

            # Main loop - process candidate centers
            N = len(core_nodes)
            covered_nodes = set()
            new_modules = []

            for x, deg, nbhd in candidates:

                # Note: nbhd includes the center
                S = len(nbhd & covered_nodes)
                c = len(nbhd)
                m = len(covered_nodes)
                pval = hypergeom.sf(S - 1, N, c, m)
                if verbose >= 2:
                    print("    CANDIDATE %s (%d)" % (x, deg))
                    print("       N=%d S=%d c=%d m=%d pval=%.2e" %
                          (N, S, c, m, pval))
                if pval > alpha and S / c <= 0.7:
                    nbhd.add(x)
                    mod_id = _insert_module(nbhd, x)
                    new_modules.append(mod_id)
                    covered_nodes |= nbhd
                    if verbose >= 2:
                        print("       ACCEPTED")

            if verbose >= 1:
                print("    PROCESSED CANDIDATES (%d modules)" %
                      len(new_modules))

            # Assign periphery and core nodes that were not covered
            remaining = set(cmpn) - covered_nodes
            if verbose >= 2:
                print("    %d nodes remaining" % len(remaining))

            while remaining:
                new_covered = set()
                for y in list(remaining):
                    nbhd = set(z for z in U[y] if z in covered_nodes)
                    if nbhd:
                        for mod_id in new_modules:
                            mmbr_set = M.nodes[mod_id].members
                            mmbr_ids = set(node.node_id for node in mmbr_set)
                            if nbhd & mmbr_ids:
                                mmbr_set.add(U.nodes[y])
                        remaining.remove(y)
                        new_covered.add(y)
                if verbose >= 2:
                    print("    ADDING %d nodes" % len(new_covered))
                covered_nodes |= new_covered

            if verbose >= 1:
                print("    ASSIGNED PERIPHERY")

            # Compute overlaps between modules
            for i, mod1 in enumerate(new_modules):
                for mod2 in new_modules[i + 1:]:
                    overlap = M.nodes[mod1].members & M.nodes[mod2].members
                    if len(overlap) > 0:
                        edge = M.insert_edge(mod1, mod2)
                        edge.overlap = overlap
                        edge.num_overlaps = len(overlap)

            if verbose >= 1:
                print("    COMPUTED OVERLAP")

    return M


def write_modnet(M, out_dir, out_prefix, nodes_map):
    """
    Write the network of modules to nodes and edges files
    """

    node_file = os.path.join(out_dir, out_prefix + '.mod.nl.txt')
    edge_file = os.path.join(out_dir, out_prefix + '.mod.el.txt')

    nodes = sorted(M.nodes.values(), key=lambda x: x.node_id)
    edges = sorted(M.edges.values(), key=lambda x: len(x.overlap))

    with open(node_file, 'w', encoding='utf-8') as fp:
        for node in nodes:
            node.write_line(fp, nodes_map)
    with open(edge_file, 'w', encoding='utf-8') as fp:
        for edge in edges:
            edge.write_line(fp, nodes_map)


def find_gcmdiff_cutoff(gcmdiff_file, L2diff_cutoff=20):
    """
    Compute an absolute correlation cutoff from differences in L2 distances
    between graphlet correlation matrices
    """
    data = []
    with open(gcmdiff_file, 'r') as fp:
        next(fp)
        for line in fp:
            r, l1, l2 = tuple(map(float, line.strip().split('\t')))
            data.append((r, l2 < L2diff_cutoff))
    data.sort(reverse=True)
    cutoff = data[-1][0]
    for i, x in enumerate(data):
        if all(y[1] for y in data[i:]):
            cutoff = x[0]
            break
    return cutoff


def find_min_gcmdiff(gcmdiff_file):
    """
    Compute an absolute correlation cutoff from differences in L2 distances
    between graphlet correlation matrices - based on global minimum
    """
    data = []
    with open(gcmdiff_file, 'r') as fp:
        next(fp)
        for line in fp:
            try:
                r, l2, l1 = tuple(map(float, line.strip().split('\t')))
                data.append((r, l2))
            except ValueError:
                pass
    cutoff = min(data, key=itemgetter(1))[0]
    return cutoff


def read_modnet_info(modnet_info_file):
    """
    Read module network metadata
    """

    list_keys = ('el_columns', 'nl_columns')
    float_keys = ('L2diff_cutoff', 'abscorr_cutoff', 'tau')
    int_keys = ('min_mod_size', )
    filepath_keys = ('corr_file', 'psannot_file', 'gcmdiff_file')

    modnet_info = {}
    info_dir = os.path.dirname(modnet_info_file)
    with open(modnet_info_file, 'r') as fp:
        for line in fp:
            key, val = line.strip().split('\t')
            if key in list_keys:
                val = val.split(',')
            elif key in float_keys:
                val = float(val)
            elif key in int_keys:
                val = int(val)
            elif key in filepath_keys:
                val = os.path.abspath(os.path.join(info_dir, val))
            modnet_info[key] = val
    return modnet_info


def do_gcmdiff_cutoff(args):
    if args.L2diff_cutoff > 0.0:
        cutoff = find_gcmdiff_cutoff(args.gcmdiff_file, args.L2diff_cutoff)
    else:
        cutoff = find_min_gcmdiff(args.gcmdiff_file)
    prefix = os.path.basename(args.gcmdiff_file).replace('.gcmdiff.txt', '')
    print("%s\t%.3f" % (prefix, cutoff))


def do_corr_filter(args):
    corr_file = args.corr_file
    psannot_file = args.psannot_file
    selected_genes = set(args.selected_genes)

    # Extract probesets from the annotation file
    #   First get the mapping of probesets to rows as used by modules
    nodes_data, nodes_map = load_corr_nodes(psannot_file)
    gene_func = lambda k: nodes_map[k]

    selected_rows = []
    for item in corr_lines_iter(corr_file, nodes_data, args.abscorr_cutoff):
        src_id, tgt_id, sr, sp = item
        src = gene_func(src_id)
        tgt = gene_func(tgt_id)
        if src not in selected_genes and tgt not in selected_genes:
            continue
        if src in selected_genes:
            src, tgt = tgt, src
        selected_rows.append((src, tgt, sr, sp, np.abs(sr)))

    selected_rows.sort(key=lambda item: item[4], reverse=True)

    header = ['GeneA', 'GeneB', 'Corr', 'CorrPvalue', 'AbsCorr']
    with io.open(args.output_file, 'w') as fp:
        with io.open(corr_file, 'r') as fp_in:
            for line in fp_in:
                if line[0] == '#':
                    fp.write(line)
                else:
                    break
        fp.write('# Correlation filter cutoff: %.4f\n' % args.abscorr_cutoff)
        fp.write('#%s\n' % '\t'.join(header))
        fmt = '%s\t%s\t%.4f\t%.2e\t%.4f\n'
        for item in selected_rows:
            fp.write(fmt % tuple(item))


def do_modnet(args):

    min_mod_size = 6
    tau = 1.0

    out_dir = args.output_dir
    out_prefix = args.output_prefix
    dataset_prefix = os.path.basename(args.corr_file).replace('.corr.txt', '')

    if args.gcmdiff_file is None:
        cutoff = args.cutoff
    elif args.L2diff_cutoff > 0.0:
        cutoff = find_gcmdiff_cutoff(args.gcmdiff_file, args.L2diff_cutoff)
    else:
        cutoff = find_min_gcmdiff(args.gcmdiff_file)

    G, nodes_map = load_corr_edges(args.corr_file, args.psannot_file, cutoff)
    M = modnet(G, min_mod_size=min_mod_size, tau=tau, verbose=args.verbose,
               consider_core_only=args.consider_core_only, steps=args.steps,
               dataset_prefix=dataset_prefix)

    write_modnet(M, out_dir, out_prefix, nodes_map)

    # Write info file with all metadata and parameters
    info_file = os.path.join(out_dir, out_prefix + '.mod.info.txt')
    with open(info_file, 'w') as fp:
        fp.write('dataset_prefix\t%s\n' % dataset_prefix)
        fp.write('corr_file\t%s\n' % os.path.relpath(args.corr_file, out_dir))
        fp.write('psannot_file\t%s\n' %
                 os.path.relpath(args.psannot_file, out_dir))
        if args.gcmdiff_file is not None:
            fp.write('gcmdiff_file\t%s\n' %
                     os.path.relpath(args.gcmdiff_file, out_dir))
        fp.write('abscorr_cutoff\t%.4f\n' % cutoff)
        fp.write('min_mod_size\t%d\n' % min_mod_size)
        fp.write('tau\t%.4f\n' % tau)
        fp.write('el_columns\t%s\n' % ','.join(ModuleLink.header))
        fp.write('nl_columns\t%s\n' % ','.join(ModuleNode.header))


def do_modnet_enrich_weights(args):
    modnet_info = read_modnet_info(args.modnet_info_file)

    gmt_path = args.modnet_info_file.replace('.mod.info.txt', '.modules.gmt')
    nl_path = args.modnet_info_file.replace('.mod.info.txt', '.mod.nl.txt')

    # enrich_dir = os.path.join(os.path.dirname(gmt_path), '%s.enrich' %
    #                           modnet_info['dataset_prefix'])
    # weights_dir = os.path.join(enrich_dir, 'weights')
    # os.makedirs(weights_dir, mode=0o777, exist_ok=True)

    all_genes = set()
    with open(nl_path, 'r') as fp_in:
        with open(gmt_path, 'w') as fp_out:
            for line in fp_in:
                fields = line.strip().split('\t')
                mod_id = fields[0]
                desc = mod_id.replace(':', '-')
                genes = fields[3].split(',')
                all_genes.update(genes)
                if len(genes) < args.min_genes:
                    continue
                fp_out.write('%s\t%s\t%s\n' % (mod_id, desc,
                                               '\t'.join(genes)))

    # with open(nl_path, 'r') as fp_in:
    #     for line in fp_in:
    #         fields = line.strip().split('\t')
    #         mod_id = fields[0]
    #         mod_filename = mod_id.replace(':', '-') + '.txt'
    #         genes = fields[3].split(',')
    #         if len(genes) < args.min_genes:
    #             continue
    #         weights_path = os.path.join(weights_dir, mod_filename)
    #         with open(weights_path, 'w') as fp_out:
    #             for g in sorted(genes):
    #                 fp_out.write('%s\t1.0\n' % g)
    #             for g in sorted(all_genes - set(genes)):
    #                 fp_out.write('%s\t0.0\n' % g)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Module networks')
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

    parser4 = subparsers.add_parser('show-gcmdiff-cutoff',
                                    help='Show correlation cutoff based on gcmdiff')
    parser4.set_defaults(func=do_gcmdiff_cutoff)
    parser4.add_argument('gcmdiff_file', help='GCM differences file')
    parser4.add_argument('-l',
                         action='store',
                         default=20,
                         type=float,
                         dest='L2diff_cutoff',
                         help='L2diff cutoff (set to 0 to ignore)')

    # ******************************************************************

    parser5 = subparsers.add_parser('corr-gene-filter',
                                    help='Filter correlations by selected genes')
    parser5.set_defaults(func=do_corr_filter)
    parser5.add_argument('corr_file', help='correlations file')
    parser5.add_argument('psannot_file', help='probeset annotations file')
    parser5.add_argument('output_file', help='output file')
    parser5.add_argument('selected_genes', help='selected_genes', nargs='+')
    parser5.add_argument('--abs-corr-cutoff', '-c',
                         action='store',
                         default=0.0,
                         type=float,
                         dest='abscorr_cutoff',
                         help='Absolute correlation cutoff')


    # ******************************************************************

    parser6 = subparsers.add_parser('modnet',
                                    help='Construct module network')
    parser6.set_defaults(func=do_modnet)
    parser6.add_argument('corr_file', help='correlations file')
    parser6.add_argument('psannot_file', help='node annotations file')
    parser6.add_argument('output_dir', help='output directory')
    parser6.add_argument('output_prefix', help='output prefix')
    parser6.add_argument('-g',
                         action='store',
                         default=None,
                         dest='gcmdiff_file',
                         help='GCM differences file')
    parser6.add_argument('-l',
                         action='store',
                         default=20.0,
                         type=float,
                         dest='L2diff_cutoff',
                         help='L2diff cutoff')
    parser6.add_argument('-c',
                         action='store',
                         default=0.0,
                         type=float,
                         dest='cutoff',
                         help='Absolute correlation cutoff')
    parser6.add_argument('--alpha',
                         action='store',
                         default=0.01,
                         type=float,
                         dest='alpha',
                         help='Hypergeometric p-value cutoff')
    parser6.add_argument('--do-not-separate-periphery', '-n',
                         action='store_false',
                         default=True,
                         dest='consider_core_only',
                         help='Do not exclude periphery when looking for module centers')
    parser6.add_argument('--steps', '-k',
                         action='store',
                         default=1,
                         type=int,
                         dest='steps',
                         help='Neighborhood steps')

    # ******************************************************************

    parser7 = subparsers.add_parser('modnet-enrich-weights',
                                    help='Construct module gmt files for enrichment')
    parser7.set_defaults(func=do_modnet_enrich_weights)
    parser7.add_argument('modnet_info_file', help='.mod.info.txt file')
    parser7.add_argument('-m',
                         action='store',
                         default=10,
                         type=int,
                         dest='min_genes',
                         help='Minimum number of genes for a module to be considered')

    # ******************************************************************

    args = parser.parse_args()
    args.func(args)
