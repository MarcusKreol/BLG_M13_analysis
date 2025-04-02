# BLG-M13 kinship analysis

Source code used in kinship analysis for BLG-M13 (2025).


## kinship_structure
Kinship structure estimation of BLG-M13 based on pedigree simulation. Related to supplementary note 3.

The major pipeline is organized into 3 wrapper notebooks:
- `pop_gen_run_pedsim.ipynb`: Running pedigree simulations and saving IBD matrix datasets.
- `summary_centrality.ipynb`: Centrality estimation. Contains source code for Fig. 4F, S13B and S13C.
- `summary_individ_num.ipynb`: Inividual number estimation. Contains source code for Fig. 4G, and S13D.

In this project, [ped-sim](https://github.com/williamslab/ped-sim) ([Caballero et al. (2019)](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1007979)) is used for pedigree simulation. Functions from [ancIBD](https://github.com/hringbauer/ancIBD) ([Ringbauer et al. (2023)](https://www.nature.com/articles/s41588-023-01582-w)) were adopted for IBD segment file processing.


## 2nd_degree_classify
Classification of 2nd-degree relationships in BLG-M13 based on multi-way IBD sharing. Related to supplementary note 2.

A rebuild base on the basic principles of CREST ([Qiao et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0002929720304407))

The major pipeline is organized into 2 wrapper notebooks:
- `crest_rebuild_test.ipynb`: Mutual relative IBD sharing calculations on BLG-M13 data. Contains source code for Fig. 4C.
- `./validation_pedsim/validation_pedsim.ipynb`: Validation of the method on artificially generated pedigrees. Contains source code for Fig. S12B.

