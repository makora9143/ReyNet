# Reynolds Equivariant and Invariant Networks


## TODO
- MLPもチャンネル方向と集合方向でたたみ込むべき
  - 現在はset_layersのみ
  - channel_layersとset_layersの両方を用意したほうがいい




## Experimental Design


- Synthetic Data Regression
  - Dataset
    - Symmetry(eq)
    - Diagonal(eq)
    - Power(eq)
    - Trace(inv)
  - Baseline
    - Maron et al.(2018)

- Graph Benchmark Classification
  - Dataset
    - MUTAG
    - IMDB-B
    - IMDB-M
    - *PTC
    - *PROTEINS
    - *NCI1
    - *NCI109
    - *COLLAB
  - Baseline
    - Maron et al.(2018)
    - DeepGraphConvNN
 MUTAG PTC PROTEINS NCI1 NCI109 COLLAB IMDB-B IMDB-M
84.61±10 59.47±7.3 75.19±4.3 73.71±2.6 72.48±2.5 77.92±1.7 71.27±4.5 48.55±3.9

- Point Cloud Classification
  - Dataset
    - ModelNet40
  - Baseline
    - DeepGraphConvNN


Deepset 5000 92 +-3
        100  82+-2
