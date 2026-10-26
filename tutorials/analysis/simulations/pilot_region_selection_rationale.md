# Pilot region selection rationale

Stratified sampling over depth (leadfield-norm tertiles) x orientation (radiality tertiles), 446 candidate cortical regions with dipoles in `sub-simf10`'s inversion source space (of 473 total atlas regions -- 27 subcortical/thalamic/brainstem regions have zero dipoles in this surface-only source space and are excluded, see orientation report).

| depth_bin | orientation_bin | n_candidates | n_selected |
|---|---|---|---|
| shallow | tangential | 36 | 2 |
| shallow | oblique | 46 | 3 |
| shallow | radial | 67 | 3 |
| mid | tangential | 64 | 4 |
| mid | oblique | 46 | 3 |
| mid | radial | 38 | 2 |
| deep | tangential | 49 | 3 |
| deep | oblique | 56 | 3 |
| deep | radial | 44 | 2 |

(Top-up: added `ctx-rh-insula_1` so lobe `insula` (0 picks from depth x orientation stratification alone) has >=1 representative.)


Total selected: 26 regions.

## Lobe distribution of selection

lobe
frontal      11
parietal      9
temporal      4
occipital     1
insula        1


## Selected regions

```
                          region      lobe hemi      depth  orientation depth_bin orientation_bin  n_dipoles
   ctx-rh-lateralorbitofrontal_1   frontal   rh  761.16595     0.330577   shallow      tangential          7
     ctx-rh-posteriorcingulate_2  parietal   rh 1000.73670     0.402735   shallow      tangential         21
    ctx-lh-medialorbitofrontal_5   frontal   lh  861.82810     0.412362   shallow         oblique          7
     ctx-rh-posteriorcingulate_1  parietal   rh  972.33990     0.498331   shallow         oblique         16
       ctx-rh-inferiortemporal_5  temporal   rh 1056.94910     0.415703   shallow         oblique         26
    ctx-rh-medialorbitofrontal_1   frontal   rh 1060.86610     0.517559   shallow          radial         13
                 ctx-rh-insula_1    insula   rh  989.40540     0.606321   shallow          radial         13
        ctx-rh-parahippocampal_1  temporal   rh  892.21075     0.768208   shallow          radial         11
       ctx-rh-inferiortemporal_1  temporal   rh  742.99316     0.526421   shallow          radial         21
       ctx-rh-superiorfrontal_12   frontal   rh 1159.68030     0.403367       mid      tangential         15
            ctx-lh-postcentral_5  parietal   lh 1189.24670     0.145968       mid      tangential         12
            ctx-rh-postcentral_1  parietal   rh 1075.47610     0.264768       mid      tangential         15
      ctx-rh-inferiorparietal_10  parietal   rh 1239.14040     0.304270       mid      tangential         15
            ctx-rh-precentral_15   frontal   rh 1220.62510     0.415983       mid         oblique         17
              ctx-lh-precuneus_1  parietal   lh 1082.22800     0.508572       mid         oblique          8
               ctx-lh-fusiform_3  temporal   lh 1069.02760     0.420461       mid         oblique         19
ctx-rh-caudalanteriorcingulate_1   frontal   rh 1075.28980     0.515543       mid          radial          9
              ctx-rh-precuneus_8  parietal   rh 1188.21340     0.644759       mid          radial         25
  ctx-rh-rostralmiddlefrontal_12   frontal   rh 1262.21290     0.407386      deep      tangential         10
             ctx-rh-precentral_5   frontal   rh 1272.22490     0.123943      deep      tangential          7
             ctx-lh-precentral_6   frontal   lh 1480.18900     0.344286      deep      tangential          8
       ctx-rh-superiorfrontal_17   frontal   rh 1292.05140     0.422353      deep         oblique         27
       ctx-lh-lateraloccipital_8 occipital   lh 1265.97100     0.498695      deep         oblique         21
          ctx-rh-supramarginal_3  parietal   rh 1378.75620     0.501508      deep         oblique         24
        ctx-rh-superiorfrontal_6   frontal   rh 1401.10330     0.508775      deep          radial         13
      ctx-rh-superiorparietal_13  parietal   rh 1485.74400     0.749149      deep          radial         24
```