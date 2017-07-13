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
import tempfile
import pickle
from subprocess import Popen, PIPE
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


GCM_ORDER = [0, 2, 5, 7, 8, 10, 11, 6, 9, 4, 1]


matplotlib.rc('font', **{'sans-serif': 'Arial',
                         'family': 'sans-serif'})
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def save_object(obj, storage_path, file_rootname, ext='.pkl'):
    """
    Stores object as a pickle.
    """
    filename = os.path.join(storage_path, '%s%s' % (file_rootname, ext))
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp, -1)


def restore_object(storage_path, file_rootname, ext='.pkl'):
    """
    Retrieves object from a pickle.
    """
    filename = os.path.join(storage_path, '%s%s' % (file_rootname, ext))
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def _normalize(X):
    Xn = np.subtract(X, X.mean(1)[:, np.newaxis])
    np.divide(Xn, np.sqrt(np.sum(Xn * Xn, 1))[:, np.newaxis], Xn)
    return Xn


def pearson(X, Y):
    """
    All-against-all Pearson correlations for data matrices X and Y
    """
    Xn = _normalize(X)
    Yn = _normalize(Y)
    R = np.dot(Xn, Yn.T)
    R = np.maximum(np.minimum(R, 1.0, R), -1.0)
    return R


def spearman(X, Y):
    """
    All-against-all Spearman correlations for data matrices X and Y
    """
    Xr = np.apply_along_axis(rankdata, 1, X)
    Yr = np.apply_along_axis(rankdata, 1, Y)
    return pearson(Xr, Yr)


def run_orca(el, orca_prefix='', graphlet_size=4):

    command = os.path.join(orca_prefix, 'orca')

    # Convert labels to consecutive integers
    used_labels = sorted(set(el.Source) | set(el.Target))
    label_map = dict((lbl, i) for i, lbl in enumerate(used_labels))

    # Create temporary files
    fd1, in_file = tempfile.mkstemp()
    fd2, out_file = tempfile.mkstemp()

    # Fill input file
    with open(in_file, 'w') as fp:
        fp.write('%d %d\n' % (len(used_labels), el.shape[0]))
        for _, src, tgt in el.itertuples():
            fp.write('%d %d\n' % (label_map[src], label_map[tgt]))

    # Run orca
    full_args = [command] + ['%d' % graphlet_size, in_file, out_file]
    FNULL = open(os.devnull, 'w')
    proc = Popen(full_args, bufsize=0, stdin=None, stdout=FNULL,
                 stderr=PIPE, universal_newlines=True)
    err_msg = proc.stderr.read().strip()
    proc.stderr.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(err_msg)

    # Read results
    G = pd.read_table(out_file, header=None, sep=' ')
    G.index = used_labels

    os.close(fd1)
    os.close(fd2)
    os.remove(in_file)
    os.remove(out_file)

    return G


def load_edges(edges_file, src_col=0, tgt_col=1, corr_col=2):
    T = pd.read_table(edges_file, usecols=(src_col, tgt_col, corr_col),
                      names=('Source', 'Target', 'Corr'),
                      comment='#')
    T['AbsCorr'] = np.abs(T.Corr)
    return T


def gdv_cga(edges_tbl, min_corr=0.3, max_corr=0.92, corr_step=0.005,
            min_edges=500, max_edges=100000, max_edge_density=0.02,
            orca_prefix='', verbose=1):

    CutRecord = namedtuple('CutRecord', ['AbsCor', 'NumNodes', 'NumEdges',
                                         'EdgeDensity', 'MaxDegree',
                                         'ClusteringCoeff'])

    cuts0 = np.arange(max_corr, min_corr, -corr_step)
    gdcs = []
    gcms = []
    graph_infos = []
    E0 = 0

    for rho in cuts0:

        el = edges_tbl[edges_tbl.AbsCorr >= rho][['Source', 'Target']]

        # Don't compute graphlets for too small or too large networks
        V = len(set(el.Source) | set(el.Target))
        if V == 0:
            continue
        E = el.shape[0]
        edensity = 2 * E / V / (V - 1)

        rho_info = "%.4f V=%d E=%d edens=%.2g" % (rho, V, E, edensity)
        if E0 >= edges_tbl.shape[0]:
            break
        E0 = E
        if E < min_edges or E > max_edges or edensity >= max_edge_density:
            if verbose > 0:
                print("%s [Skipping]" % rho_info)
            continue
        elif verbose > 0:
            print(rho_info)

        GDC = run_orca(el, orca_prefix)
        gdcs.append(GDC)
        max_deg = GDC[[0]].values.max()

        # Calculate clustering coefficient
        G2 = GDC[[3]].values.sum() / 3
        G1 = GDC[[2]].values.sum() / 2 + GDC[[1]].values.sum()
        ccoef = G2 / (G2 + G1)
        graph_infos.append(CutRecord._make((rho, V, E, edensity, max_deg,
                                           ccoef)))

        # Calculate correlations between orbits
        GDC1 = GDC.iloc[:, GCM_ORDER].values.T

        GCM = spearman(GDC1, GDC1)
        GCM[np.isnan(GCM)] = 0.0
        gcms.append(GCM)

    ginfo = pd.DataFrame.from_records(graph_infos, columns=CutRecord._fields)
    return gdcs, gcms, ginfo


def gcm_heatmaps(gcms, ginfo, out_file, title_tag, colormap='RdYlBu_r'):

    with PdfPages(out_file) as pdf:
        font0 = FontProperties()
        font0.set_name('sans-serif')
        font0.set_size(12)
        font1 = font0.copy()
        font1.set_size(14)
        font2 = font0.copy()
        font2.set_size(16)

        for i, GCM in enumerate(gcms):

            title = "%s (|cor|> %.3f, %d nodes, %d edges)" % \
                    (title_tag, ginfo.loc[i, 'AbsCor'],
                     ginfo.loc[i, 'NumNodes'],
                     ginfo.loc[i, 'NumEdges'])

            fig = plt.figure(1, figsize=(8, 8))
            ax = fig.add_subplot(111)
            im = ax.matshow(GCM, aspect='equal',
                            cmap=matplotlib.cm.get_cmap(colormap),
                            vmin=-1.0, vmax=1.0)
            ax.set_title(title, fontproperties=font2)

            labels = list(map(str, GCM_ORDER))
            xticks = np.arange(len(labels))
            yticks = np.arange(len(labels))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(labels, fontproperties=font1)
            ax.set_yticklabels(labels, fontproperties=font1)

            plt.setp(list(ax.spines.values()), lw=0.5, color="#666666")
            plt.setp(ax.get_xticklabels(), fontproperties=font1)
            plt.setp(ax.get_yticklabels(), fontproperties=font1)
            plt.setp(ax.xaxis.get_ticklines(), markersize=3)
            plt.setp(ax.yaxis.get_ticklines(), markersize=3)
            plt.setp(ax.xaxis.get_ticklines(minor=True), markersize=1)
            plt.setp(ax.yaxis.get_ticklines(minor=True), markersize=1)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('right')

            pdf.savefig()
            plt.close()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_diffgcm(gcms, ginfo, window_size=5):

    dcorr = -np.diff(ginfo.loc[:, 'AbsCor'].values)
    diff_gcm = [gcms[i] - gcms[i - 1] for i in range(1, len(gcms))]
    l1_diff_gcm = np.array([np.abs(D).sum() for D in diff_gcm],
                           dtype=np.float64) / dcorr
    l2_diff_gcm = np.array([np.sqrt((D * D).sum()) for D in diff_gcm],
                           dtype=np.float64) / dcorr

    n = 1
    if window_size is not None:
        n = window_size
        l1_diff_gcm = moving_average(l1_diff_gcm, n)
        l2_diff_gcm = moving_average(l2_diff_gcm, n)

    df = pd.DataFrame(ginfo.iloc[n: ginfo.shape[0], :])
    df['L1diffGCM'] = l1_diff_gcm
    df['L2diffGCM'] = l2_diff_gcm
    return df


def gcmdiff_plots(tbl, out_file):

    with PdfPages(out_file) as pdf:
        font0 = FontProperties()
        font0.set_name('sans-serif')
        font0.set_size(8)
        font1 = font0.copy()
        font1.set_size(9)
        font2 = font0.copy()
        font2.set_size(11)

        fig = plt.figure(1, figsize=(8, 11))
        for i, v in enumerate(tbl.columns.values[1:4]):
            ax = fig.add_subplot(3, 1, i+1)
            p = ax.plot(tbl.loc[:,'AbsCor'].values, tbl.loc[:,v].values, 'k-o')
            title = v
            ax.set_title(title, fontproperties=font2)
            ax.invert_xaxis()

            plt.setp(list(ax.spines.values()), lw=0.5, color="#666666")
            plt.setp(ax.get_xticklabels(), fontproperties=font1)
            plt.setp(ax.get_yticklabels(), fontproperties=font1)
            plt.setp(ax.xaxis.get_ticklines(), markersize=3)
            plt.setp(ax.yaxis.get_ticklines(), markersize=3)
            plt.setp(ax.xaxis.get_ticklines(minor=True), markersize=1)
            plt.setp(ax.yaxis.get_ticklines(minor=True), markersize=1)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.close()
        pdf.savefig(fig)

        fig = plt.figure(1, figsize=(8, 11))
        for i, v in enumerate(tbl.columns.values[4:6]):
            ax = fig.add_subplot(3, 1, i+1)
            p = ax.plot(tbl.loc[:,'AbsCor'].values, tbl.loc[:,v].values, 'k-o')
            title = v
            ax.set_title(title, fontproperties=font2)
            ax.invert_xaxis()

            plt.setp(list(ax.spines.values()), lw=0.5, color="#666666")
            plt.setp(ax.get_xticklabels(), fontproperties=font1)
            plt.setp(ax.get_yticklabels(), fontproperties=font1)
            plt.setp(ax.xaxis.get_ticklines(), markersize=3)
            plt.setp(ax.yaxis.get_ticklines(), markersize=3)
            plt.setp(ax.xaxis.get_ticklines(minor=True), markersize=1)
            plt.setp(ax.yaxis.get_ticklines(minor=True), markersize=1)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.close()
        pdf.savefig(fig)

        fig = plt.figure(1, figsize=(8, 11))
        for i, v in enumerate(tbl.columns.values[6:8]):
            ax = fig.add_subplot(3, 1, i+1)
            p = ax.plot(tbl.loc[:,'AbsCor'].values, tbl.loc[:,v].values, 'k-o')
            title = v
            ax.set_title(title, fontproperties=font2)
            ax.invert_xaxis()

            plt.setp(list(ax.spines.values()), lw=0.5, color="#666666")
            plt.setp(ax.get_xticklabels(), fontproperties=font1)
            plt.setp(ax.get_yticklabels(), fontproperties=font1)
            plt.setp(ax.xaxis.get_ticklines(), markersize=3)
            plt.setp(ax.yaxis.get_ticklines(), markersize=3)
            plt.setp(ax.xaxis.get_ticklines(minor=True), markersize=1)
            plt.setp(ax.yaxis.get_ticklines(minor=True), markersize=1)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.close()
        pdf.savefig(fig)


def do_run(args):

    out_dir = args.gca_dir
    out_prefix = args.dataset_tag
    edges_tbl = load_edges(args.corr_file, args.src_col, args.tgt_col,
                           args.corr_col)

    kwargs = vars(args)
    for k in ('corr_file', 'src_col', 'tgt_col', 'corr_col', 'gca_dir',
              'dataset_tag', 'func'):
        kwargs.pop(k)

    gdcs, gcms, ginfo = gdv_cga(edges_tbl, **kwargs)
    save_object((gdcs, gcms, ginfo), out_dir, out_prefix)


def do_gcm_heatmaps(args):

    out_dir = args.gca_dir
    out_prefix = args.dataset_tag
    obj = restore_object(out_dir, out_prefix)
    if len(obj) == 3:
        gdcs, gcms, ginfo = restore_object(out_dir, out_prefix)
    else:
        gdcs, gcms, ginfo, gcms_partial = restore_object(out_dir, out_prefix)
    pdf_file = os.path.join(out_dir, out_prefix + '.hmap.pdf')
    gcm_heatmaps(gcms, ginfo, pdf_file, out_prefix, args.colormap)

    if len(obj) > 3:
        pdf_file = os.path.join(out_dir, out_prefix + '.hmap2.pdf')
        gcm_heatmaps(gcms_partial, ginfo, pdf_file, out_prefix, args.colormap)


def do_gcmdiff(args):

    out_dir = args.gca_dir
    out_prefix = args.dataset_tag
    gdcs, gcms, ginfo = restore_object(out_dir, out_prefix)
    tbl = compute_diffgcm(gcms, ginfo, args.window_size)
    gcmdiff_file = os.path.join(out_dir, out_prefix + '.gcmdiff.txt')
    tbl1 = tbl[['AbsCor', 'L1diffGCM', 'L2diffGCM']]
    tbl1.to_csv(gcmdiff_file, index=None, sep='\t')

    plot_file = os.path.join(out_dir, out_prefix + '.gcmdiff.pdf')
    gcmdiff_plots(tbl, plot_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Graphlet Correlation Analysis')
    subparsers = parser.add_subparsers()

    parser1 = subparsers.add_parser('run',
                                    help='Compute graphlets for a range of cutoffs')

    parser1.set_defaults(func=do_run)
    parser1.add_argument('corr_file', help='correlations file')
    parser1.add_argument('gca_dir', help='output directory')
    parser1.add_argument('dataset_tag', help='output file prefix')

    parser1.add_argument('--src-col',
                         action='store',
                         default=0,
                         dest='src_col',
                         type=int,
                         help='Edge source column')
    parser1.add_argument('--tgt-col',
                         action='store',
                         default=1,
                         dest='tgt_col',
                         type=int,
                         help='Edge target column')
    parser1.add_argument('--corr-col',
                         action='store',
                         default=2,
                         dest='corr_col',
                         type=int,
                         help='Edge correlation column')
    parser1.add_argument('-v',
                         action='store',
                         default=1,
                         type=int,
                         dest='verbose',
                         help='Verbosity level')
    parser1.add_argument('--min-corr',
                         action='store',
                         default=0.2,
                         dest='min_corr',
                         type=float,
                         help='Minimum correlation to be considered')
    parser1.add_argument('--max-corr',
                         action='store',
                         default=0.95,
                         dest='max_corr',
                         type=float,
                         help='Maximum correlation to be considered')
    parser1.add_argument('--corr-step',
                         action='store',
                         default=0.005,
                         dest='corr_step',
                         type=float,
                         help='Correlation step')
    parser1.add_argument('--min-edges',
                         action='store',
                         default=500,
                         dest='min_edges',
                         type=int,
                         help='Minimum edges for a step')
    parser1.add_argument('--max-edges',
                         action='store',
                         default=100000,
                         dest='max_edges',
                         type=int,
                         help='Maximum edges for a step')
    parser1.add_argument('--max-edge-density',
                         action='store',
                         default=0.02,
                         dest='max_edge_density',
                         type=float,
                         help='Maximum edge density')
    parser1.add_argument('--orca-path',
                         action='store',
                         default='',
                         dest='orca_prefix',
                         help='ORCA path')

    parser2 = subparsers.add_parser('heatmaps',
                                    help='Plot GCM heatmaps in a pdf document')

    parser2.set_defaults(func=do_gcm_heatmaps)
    parser2.add_argument('gca_dir', help='gca directory')
    parser2.add_argument('dataset_tag', help='dataset prefix')
    parser2.add_argument('--colormap', '-c',
                         action='store',
                         default='RdYlBu_r',
                         dest='colormap',
                         help='Matplotlib colormap')

    parser3 = subparsers.add_parser('gcmdiff',
                                    help='Compute and plot differences between GCMs')

    parser3.set_defaults(func=do_gcmdiff)
    parser3.add_argument('gca_dir', help='gca directory')
    parser3.add_argument('dataset_tag', help='dataset prefix')
    parser3.add_argument('--window-size', '-w',
                         action='store',
                         default=None,
                         type=int,
                         dest='window_size',
                         help='Moving average window size')

    args = parser.parse_args()
    args.func(args)
