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

PYTHON := python3

ORIGDATADIR := data
DATADIR1 := data-ranked
OUTPUTDIR1 := out
GCADIR := $(OUTPUTDIR1)/gca
MODULEDIR := modules
VIRTUALENVDIR := venv
VENVPY := $(VIRTUALENVDIR)/bin/python3

CODEDIR := code
STYLEDIR := styles
ORCAPATH := orca

modnet_style := 'Modnet03.sty.json'
genenet_style := 'Coexpnet04.sty.json'


# ------------------------------------------------------------------------------------
# *** CHANGE ONLY HERE ***
# ------------------------------------------------------------------------------------

dataset_prefixes := GSE57945-Ileum-Infl GSE57945-Ileum-NotCD WashU_CohortA-Ileum-All

# ------------------------------------------------------------------------------------
# *** END OF CHANGES ***
# ------------------------------------------------------------------------------------

data_files := $(foreach prefix,$(dataset_prefixes),$(ORIGDATADIR)/$(prefix).expr.txt.gz $(DATADIR1)/$(prefix).expr.txt.gz  $(ORIGDATADIR)/$(prefix).psannot.txt $(DATADIR1)/$(prefix).psannot.txt)
corr_files := $(foreach prefix,$(dataset_prefixes),$(OUTPUTDIR1)/$(prefix).corr.txt)
gca_files := $(foreach prefix,$(dataset_prefixes),$(GCADIR)/$(prefix).pkl)
gcmdiff_files := $(foreach prefix,$(dataset_prefixes),$(GCADIR)/$(prefix).gcmdiff.txt $(GCADIR)/$(prefix).hmap.pdf)
modnet_files := $(foreach prefix,$(dataset_prefixes),$(MODULEDIR)/$(prefix).mod.el.txt)
modinfo_files := $(foreach prefix,$(dataset_prefixes),$(MODULEDIR)/$(prefix).mod.info.txt)
gmt_files := $(foreach prefix,$(dataset_prefixes),$(MODULEDIR)/$(prefix).modules.gmt)

.PHONY: orca gca modnets gmts cytomodnets

all: $(data_files) $(corr_files) $(gca_files) $(gcmdiff_files)  $(modnet_files) $(gmt_files)

datafiles: $(data_files)

gca: $(corr_files) $(gcmdiff_files)

modnets: $(modnet_files) $(gmt_files)

$(ORCAPATH)/orca:
	make -C $(ORCAPATH)

$(VIRTUALENVDIR):
	$(PYTHON) -m venv $@
	$(VIRTUALENVDIR)/bin/pip install --upgrade pip
	$(VIRTUALENVDIR)/bin/pip install -r requirements.txt

# ------------------------------------------------------------------------------------
# Recipies for transforming data
# ------------------------------------------------------------------------------------

$(DATADIR1)/%.expr.txt.gz: $(ORIGDATADIR)/%.expr.txt.gz | $(DATADIR1) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/coexpr.py transform-dataset -trank $< $@

$(DATADIR1)/%.psannot.txt: $(ORIGDATADIR)/%.psannot.txt | $(DATADIR1) $(VIRTUALENVDIR)
	cp -a $< $@

# ------------------------------------------------------------------------------------
# Recipies for generating correlations
# ------------------------------------------------------------------------------------

$(OUTPUTDIR1)/%.corr.txt: $(DATADIR1)/%.expr.txt.gz | $(OUTPUTDIR1) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/coexpr.py -v1 auto-corr-run $< $(OUTPUTDIR1) $(*F)

# ------------------------------------------------------------------------------------
# Recipies for generating gdrop plots etc.
# ------------------------------------------------------------------------------------

$(GCADIR)/%.pkl: $(OUTPUTDIR1)/%.corr.txt $(ORCAPATH)/orca | $(GCADIR) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/gdv-cga.py run -v1 --orca-path=$(ORCAPATH) --max-corr=0.98 --max-edge-density=0.3 --min-edges=100  $< $(GCADIR) $(*F)

$(GCADIR)/%.hmap.pdf: $(GCADIR)/%.pkl | $(GCADIR) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/gdv-cga.py heatmaps $(GCADIR) $(*F)

$(GCADIR)/%.gcmdiff.txt: $(GCADIR)/%.pkl | $(GCADIR) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/gdv-cga.py gcmdiff $(GCADIR) $(*F)

# ------------------------------------------------------------------------------------
# Recipies for generating module networks
# ------------------------------------------------------------------------------------

# This is a special case of a directly chosen absolute correlation cutoff (0.75)
# used for WashU_CohortA-Ileum

$(MODULEDIR)/WashU_CohortA-Ileum-All.mod.el.txt $(MODULEDIR)/WashU_CohortA-Ileum-All.mod.nl.txt $(MODULEDIR)/WashU_CohortA-Ileum-All.mod.info.txt: $(GCADIR)/WashU_CohortA-Ileum-All.gcmdiff.txt $(OUTPUTDIR1)/WashU_CohortA-Ileum-All.corr.txt $(DATADIR1)/WashU_CohortA-Ileum-All.psannot.txt | $(MODULEDIR) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/modulenet.py modnet -k1 -c 0.75 $(OUTPUTDIR1)/WashU_CohortA-Ileum-All.corr.txt $(DATADIR1)/WashU_CohortA-Ileum-All.psannot.txt $(MODULEDIR) WashU_CohortA-Ileum-All

$(MODULEDIR)/%.mod.el.txt $(MODULEDIR)/%.mod.nl.txt $(MODULEDIR)/%.mod.info.txt: $(GCADIR)/%.gcmdiff.txt $(OUTPUTDIR1)/%.corr.txt $(DATADIR1)/%.psannot.txt | $(MODULEDIR) $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/modulenet.py modnet -k1 -l 25.0 -g $(GCADIR)/$(*F).gcmdiff.txt $(OUTPUTDIR1)/$(*F).corr.txt $(DATADIR1)/$(*F).psannot.txt $(MODULEDIR) $(*F)

# ------------------------------------------------------------------------------------
# Recipies for generating enrichment files (GMT)
# ------------------------------------------------------------------------------------

%.modules.gmt: %.mod.info.txt | $(VIRTUALENVDIR)
	$(VENVPY) $(CODEDIR)/modulenet.py modnet-enrich-weights $<

# ------------------------------------------------------------------------------------
# Recipies for importing networks into Cytoscape
#
# This requires Cytoscape 3.4 or later to be started and listening on port 1234
#
# ------------------------------------------------------------------------------------

cytomodnets: $(modnet_files) | $(VIRTUALENVDIR)
	for f in $(modnet_files) ; do \
		echo $$f; \
		$(VENVPY) $(CODEDIR)/net2cyto.py modnet -s $(STYLEDIR)/$(modnet_style) $$f ; \
	done

cytofullnets: $(modinfo_files) | $(VIRTUALENVDIR)
	for f in $(modinfo_files) ; do \
		echo $$f; \
		$(VENVPY) $(CODEDIR)/net2cyto.py modsub -s $(STYLEDIR)/$(genenet_style) $$f -; \
	done

# ------------------------------------------------------------------------------------
# Other recipies
# ------------------------------------------------------------------------------------

$(ORIGDATADIR) $(DATADIR1) $(OUTPUTDIR1) $(GCADIR) $(MODULEDIR):
	mkdir -p $@

clean:
	make -C $(ORCAPATH) clean
	-rm -rf $(DATADIR1) $(OUTPUTDIR1) $(GCADIR) $(MODULEDIR) $(VIRTUALENVDIR)
	-rm -rf $(CODEDIR)/__pycache__
