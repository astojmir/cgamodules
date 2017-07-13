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
import numpy as np
from scipy.stats import rankdata
from scipy.special import beta
from scipy.special import betainc as betai
from scipy.optimize import brentq
from numpy.random import randint

np.seterr(divide='ignore')
BLOCK_SIZE = 5000


def _corr_pval(R, df):
    t_squared = R * R * (df / ((1.0 - R) * (1.0 + R)))
    return betai(0.5 * df, 0.5, df / (df + t_squared))


def corr_pval(R, N, tol=1e-8):
    """
    Safe version of correlation p-value
    """
    P = np.zeros_like(R)
    ix = (R < 1.0 - tol) & (R > -1.0 + tol)
    P[ix] = _corr_pval(R, N - 2)[ix]
    return P


def hr_cutoff(n, p, cross_corr=False):
    """
    Hero & Rajaratnam cutoff
    """
    c_n = 2 * beta((n - 2) / 2, 0.5)
    q = p if cross_corr else (p - 1)
    rho_c = np.sqrt(1 - (c_n * q) ** (-2 / (n - 4)))
    return rho_c


def find_corr_cutoff(X, Y, pval_cutoff=1e-6, prop_cutoff=5e-04,
                     use_hr_cutoff=False, abscorr_cutoff=0.0,
                     verbose=1):
    """
    Find the absolute correlation cutoff based on several criteria.

    NOTE: we assume the datasets are already normalized for empirical
           sampling.
    """

    SSIZE_FACTOR = 100
    assert X.shape[1] == Y.shape[1]
    N = X.shape[1]

    if verbose > 0:
        print("* Selection of absolute correlation cutoff")
        print("P-value cutoff = %.4e" % pval_cutoff)

    # First cutoff is based on p-value
    cor1 = 0.0
    if pval_cutoff < 0.999:
        cor1 = brentq(lambda R: _corr_pval(R, N - 2) - pval_cutoff, 0, 0.999)
        if verbose > 0:
            print("Cutoff cor1 = %.4f (Fisher's correlation p-value)" % cor1)

    # The Second is based on sampling
    cor2 = 0.0
    if prop_cutoff < 0.999:
        M = int(SSIZE_FACTOR / prop_cutoff)
        samples_x = randint(0, X.shape[0], M)
        samples_y = randint(0, Y.shape[0], M)
        corrs = [np.abs(np.dot(X[i, :], Y[j, :]))
                 for i, j in zip(samples_x, samples_y)]
        corrs.sort()
        cor2 = corrs[-(SSIZE_FACTOR + 1)]
        if verbose > 0:
            print("Cutoff cor2 = %.4f (Empirical p-value)" % cor2)

    # Hero & Rajaratnam cutoff
    cor3 = 0.0
    if hr_cutoff:
        cross_corr = X is not Y
        cor3 = hr_cutoff(N, max(X.shape[0], Y.shape[0]), cross_corr)
        if verbose > 0:
            print("Cutoff cor3 = %.4f (Hero & Rajaratnam threshold)" % cor3)

    # Absolute correlations
    cor4 = abscorr_cutoff
    if cor4 > 0.0:
        if verbose > 0:
            print("Cutoff cor4 = %.4f (User-specified cutoff)" % cor4)

    corr_cutoff = max(cor1, cor2, cor3, cor4)
    if verbose > 0:
        print("Final cutoff = %.4f (Maximum of all cutoffs)" % corr_cutoff)
    return corr_cutoff


def get_block_bounds(N, num_blocks, k):

    r = N % num_blocks
    m = N // num_blocks
    a = k * m + r
    b = a + m
    if k == 0:
        a = 0
    return a, b


def _do_block(X, fp, corr_cutoff, a1, b1, a2, b2, diagonal=False, verbose=0):
    if verbose > 0:
        print("X-range: [%d, %d), Y-range: [%d, %d)" % (a1, b1, a2, b2))
    R1 = np.dot(X[a1:b1, :], X[a2:b2, :].T)
    R1 = np.maximum(np.minimum(R1, 1.0, R1), -1.0, R1)
    if diagonal:
        R1 = np.triu(R1, k=1)
    selected = np.abs(R1) > corr_cutoff
    item_ix = np.where(selected)
    _R1 = R1[selected]
    _P1 = corr_pval(_R1, X.shape[1])
    for k, (i, j) in enumerate(zip(*item_ix)):
        # assert a1+i < a2+j
        row = ('%d' % (a1 + i), '%d' % (a2 + j),
               '%.4f' % _R1[k], '%.2e' % _P1[k])
        fp.write('\t'.join(row))
        fp.write('\n')


def autocorr_all_vs_all(X, corr_cutoff, out_dir, prefix, verbose=0):
    """
    Compute all-against-all correlations between X and
    itself and write those with absolute values above corr_cutoff to a file
    """

    os.makedirs(os.path.abspath(out_dir), exist_ok=True)

    _has_rem = int(X.shape[0] % BLOCK_SIZE > 0)
    x_num_blocks = (X.shape[0] // BLOCK_SIZE) + _has_rem
    x_blocks = [get_block_bounds(X.shape[0], x_num_blocks, i)
                for i in range(0, x_num_blocks)]
    y_blocks = x_blocks

    out_file = os.path.join(out_dir, "%s.corr.txt" % prefix)
    with open(out_file, 'w') as fp:
        fp.write('# dataset = %s\n' % prefix)
        fp.write('# num_samples = %d\n' % X.shape[1])
        fp.write('# Correlation cutoff: %.4f\n' % corr_cutoff)

        for a1, b1 in x_blocks:
            for a2, b2 in y_blocks:
                # Skipping half of pairs due to symmetry
                if a1 >= b2:
                    continue

                if b1 <= a2:
                    # No overlap between intervals
                    if verbose > 0:
                        print("No overlap")
                    _do_block(X, fp, corr_cutoff, a1, b1, a2, b2,
                              diagonal=False, verbose=verbose)
                elif a1 < a2:
                    # Block plus triangle
                    if verbose > 0:
                        print("Block plus triangle")
                    _do_block(X, fp, corr_cutoff, a1, a2, a2, b2,
                              diagonal=False, verbose=verbose)
                    _do_block(X, fp, corr_cutoff, a2, min(b1, b2), a2, b2,
                              diagonal=True, verbose=verbose)
                else:
                    # Triangle only (a1 >= a2)
                    if verbose > 0:
                        print("Triangle only")
                    _do_block(X, fp, corr_cutoff, a1, min(b1, b2), a2, b2,
                              diagonal=True, verbose=verbose)


def do_transform(args):
    X0 = np.loadtxt(args.dataset_file, delimiter='\t', ndmin=2)

    if args.transpose:
        X0 = X0.transpose()
    X = X0.copy()
    if args.transform == 'rank':
        X = np.apply_along_axis(rankdata, 1, X)
    if args.center:
        np.subtract(X, X.mean(1)[:, np.newaxis], X)
    if args.normalize:
        np.divide(X, np.sqrt(np.sum(X * X, 1))[:, np.newaxis], X)

    np.savetxt(args.output_file, X, delimiter='\t')


def do_auto_corr(args):
    X = np.loadtxt(args.dataset_file, delimiter='\t', ndmin=2)
    corr_cutoff = find_corr_cutoff(X, X, args.pval_cutoff,
                                   args.prop_cutoff,
                                   args.hr_cutoff,
                                   args.abscorr_cutoff,
                                   verbose=args.verbose)
    autocorr_all_vs_all(X, corr_cutoff, args.output_dir, args.prefix,
                        verbose=args.verbose)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Correlation networks')
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

    parser1 = subparsers.add_parser('transform-dataset',
                                    help='Transform dataset')
    parser1.set_defaults(func=do_transform)
    parser1.add_argument('dataset_file', help='input data matrix')
    parser1.add_argument('output_file', help='output data matrix')
    parser1.add_argument('-t',
                         action='store',
                         default='rank',
                         dest='transform',
                         help='Transformation to ranks')
    parser1.add_argument('-T',
                         action='store_true',
                         default=False,
                         dest='transpose',
                         help='Transpose data matrix')
    parser1.add_argument('-r', '--no-center',
                         action='store_false',
                         default=True,
                         dest='center',
                         help='Do not center each row (default is to center)')
    parser1.add_argument('-n', '--no-normalize',
                         action='store_false',
                         default=True,
                         dest='normalize',
                         help='Do not normalize each row')

    # ******************************************************************

    parser3 = subparsers.add_parser('auto-corr-run',
                                    help='Compute within-dataset correlations')
    parser3.set_defaults(func=do_auto_corr)
    parser3.add_argument('dataset_file', help='data matrix')
    parser3.add_argument('output_dir', help='output_directory')
    parser3.add_argument('prefix', help='results file prefix')
    parser3.add_argument('-a',
                         action='store_true',
                         default=False,
                         dest='hr_cutoff',
                         help='Use Hero & Rajaratnam cutoff')
    parser3.add_argument('--pval-cutoff', '-p',
                         action='store',
                         default=1e-6,
                         type=float,
                         dest='pval_cutoff',
                         help='Use Fisher P-value cutoff (1 to ignore)')
    parser3.add_argument('--prop-cutoff', '-r',
                         action='store',
                         default=5e-4,
                         type=float,
                         dest='prop_cutoff',
                         help='Empirical proportion cutoff (1 to ignore)')
    parser3.add_argument('--abs-corr-cutoff', '-c',
                         action='store',
                         default=0.0,
                         type=float,
                         dest='abscorr_cutoff',
                         help='Absolute correlation cutoff')

    # ******************************************************************

    args = parser.parse_args()
    args.func(args)
