# cga-modules
Correlation Graphlet Analysis (CGA) for construction of co-expression module
networks

## Description
This package provides code and example datasets for analyzing gene coexpression
networks using graphlet topology and for deriving networks of co-expression
modules. It also contains scripts that facilitate easy integration with
Cytoscape for network visualization and various tools for enrichment analysis.

## Installation
### Requirements
cga-modules was tested on Linux (RHEL 7) and MacOSX. It requires
* Python 3.5 or higher
* Development tools (gcc, g++, GNU Make etc.)

All other dependencies are automatically installed in a local Python virtual
environment.

### Extracting Package
Copy this package to a directory of your choice and unpack it
```sh
$ unzip cga-modules.zip
```

## Running CGA
To simplify the process of performing CGA, the steps of a standard workflow are
defined in the included `Makefile`. The package also includes
[three example datasets](#example-datasets). By default, the `Makefile` is
setup to executing CGA on these datasets. The next subsections describe the
process of performing CGA with the included datasets. The process of adding new
datasets is described [below](#adding-new-datasets).

### Computing Networks
Enter the unpacked directory and run make (replace `path/to/python3` with the
appropriate path to a Python 3.5 or higher instance).
```sh
$ cd cga-modules
$ make PYTHON=path/to/python3
```
The `make` run will first install a Python virtualenv with all dependencies and
then compute for each of the three example datasets:
* Rank-transformed datasets, in `data-ranked/` subdirectory
* A preliminary lower bound for absolute correlations between pairs of genes
* All pairs of genes with absolute Spearman correlation greater than computed
  lower bound, in `out/` subdirectory
* Measures of global network topologies for absolute correlation thresholds
  between the lower bound and 1.0, in `out/gca` subdirectory
  * This step uses *orca* tool (Hocevar & Demsar, supplied in `orca/`
    subdirectory) to compute graphlet counts for each network
* The optimal threshold for master co-expression network
* Derived co-expression module networks
  * Modules with not less than 10 genes are available as gene lists in GMT
    format

### Visualizing Networks in Cytoscape
This step requires Cytoscape, version 3.4 or later to be running on localhost
and listening to its standard port 1234. It uses Cytoscape RESTful API.

To visualize module networks run
```sh
$ make cytomodnets
```

To visualize master gene co-expression networks run
```sh
$ make cytofullnets
```

## Package Contents
### Python Code
The code necessary to run algorithms is placed in `code/` subdirectory. Please
examine individual files and `Makefile` to explore their exact functionalities
and options.

### Example Datasets
This package contains three example transcriptomic datasets, in `data/` subdirectory:
* `GSE57945-Ileum-Infl` (inflamed ileal biopsies from pediatric Crohn's Disease patients)
* `GSE57945-Ileum-NotCD` (non-inflamed ileal biopsies from pediatric subject
  without Crohn's Disease diagnosis)
* `WashU_CohortA-Ileum-All` (formalin-fixed, paraffin-embedded ileal tissue
  sections obtained from CD subjects and controls without inflammatory bowel
  disease)

Each dataset contains a data matrix, as a gzipped tab-delimited text file,
without headers, where rows represent genes and columns represent samples,
together with a gene annotation table (ending with `psannot.txt`).

### Cytoscape Styles
The `styles/` subdirectory contains the styles for visualizing the master
co-expression networks and module networks in Cytoscape.

## Adding New Datasets
* Choose a prefix for your new dataset (e.g. `prefix1`)
* Construct a gene expression data matrix as a tab-delimited text file, without
  headers, where rows represent genes and columns represent samples. Save it
  under `data/prefix1.expr.txt` (gzipping is optional)
* Produce a gene annotation table for your dataset, in tab-delimited text
  format, resembling existing examples. It should contain a header as first
  line and have at least three columns: `Probeset`, `Label`, and
  `GeneSymbol`. Save it under `data/prefix1.psannot.txt`.
* Edit `dataset_prefixes` variable in `Makefile` to include your new dataset:
```
dataset_prefixes := GSE57945-Ileum-Infl GSE57945-Ileum-NotCD WashU_CohortA-Ileum-All prefix1
```
* Re-run `make` as for examples. You may choose to run the algortihm in stages
  in order to examine the graphlet-based topologies and perhaps adjust the
  final thresholds:

```sh
$ make PYTHON=path/to/python3 gca
```
> Examine figures in `out/gca` directory here and edit Makefile

```sh
$ make modnets
```

## License and Authors
**License:** The code has been placed into public domain.
**Author:** Aleksandar Stojmirovic <astojmir@its.jnj.com>
**Author:** Patrick Kimes <patrick.kimes@gmail.com>
