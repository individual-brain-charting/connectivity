# IBC Connectivity

Code repository for connectivity analysis on IBC data and others.

## Structural Connectivity (SC) Estimation

```bash
python estimate_sc.py
```

<!-- ### Steps
1. The streamlines obtained from tractography were first warped into MNI152 space using ANTs' image registration `antsRegistration` and MRtrix's `tcktransform` in script `estimate_sc.py`.
2. In addition, the script `estimate_sc.py` also transforms the given atlas to the native individual space. This way we can calculate two kinds of structural connectivity matrices: one in the MNI space and the other in the native individual space.
3. Finally, the two connectomes are calculated using MRtrix's `tck2connectome` function in the same script `estimate_sc.py`. -->

## Functional Connectivity (FC) Estimation and Classification

```bash
python estimate_fc_classify_fc.py
```

## Similarity between FC and SC

```bash
python estimate_fc_calculate_similarity.py
```