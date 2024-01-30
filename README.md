# Reynolds Equivariant and Invariant Networks


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
  - Download: https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?dl=0
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
    - Maron et al.(2019)
    - GIN

## Prerequistics

Please Download Datasets and place them as below:

```
.
├── data
│   ├── ModelNet40
│   ├── TUDataset
│   │   ├── IMDB-BINARY
│   │   ├── IMDB-MULTI
│   │   ├── MUTAG
│   │   ├── NCI1
│   │   ├── PROTEINS
│   │   ├── PTC
│   │   └── processed
│   │── benchmark_graphs
│   │   ├── COLLAB
│   │   ├── DD
│   │   ├── ENZYMES
│   │   ├── IMDBBINARY
│   │   ├── IMDBMULTI
│   │   ├── MUTAG
│   │   ├── NCI1
│   │   ├── NCI109
│   │   ├── PROTEINS
│   │   ├── PTC
│   │   └── Synthie
│   ├── ogbg_molhiv
│   ├── ogbg_molpcba
│   └── test
│       ├── IMDBBINARY
│       └── MUTAG
└── molhiv-exp
    ├── __pycache__
    └── data
        └── ogbg_molhiv
```
