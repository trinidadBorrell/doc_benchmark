Research-Only Analysis Scripts
==============================
These scripts are NOT part of the main pipeline (cookbooks/pipeline.py).
They are standalone research tools for post-hoc analysis and figure generation.

File                                  Purpose                                              Input                                      Output
----                                  -------                                              -----                                      ------
global_analysis.py                    Cross-subject comparison plots (orig vs recon)        MARKERS results dir, patient_labels CSV    PNG plots, statistics
individual_analysis.py                Per-subject analysis with GFP                         MARKERS dir, FIF files                     Per-subject PNG plots
statistical_analysis.py               Permutation cluster tests, Wilcoxon, t-tests          MARKERS scalars/topos                      Statistical result plots
ohbm_biomarker_group_comparison.py    9x4 topoplot grid (MCS/UWS/Control LG/RS)            MARKERS topos                              Comparison PNG
ohbm_plots.py                         OHBM presentation figures (4x1 blue layout)           Decoder pickle results                     OHBM-formatted PNGs
control_rs_plots_CBraMod.py           CBraMod control RS topo comparison                    Original + CBraMod MARKERS topos           Grid topo PNGs
global_topoplots_minimal.py           10 specific topo comparison plots                     MARKERS topos, patient_labels              Named PNG files
qualitative_analysis.py               Heatmaps, time-frequency for 6 subjects               FIF files, MARKERS                         Qualitative PNGs
