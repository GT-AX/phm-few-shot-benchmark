| Method | CNN | 1 shot [%] | 5 shots [%] | 10 shots [%] | Parameters |
|---|---|---:|---:|---:|---:|
| MAML | 1D | 67.10 ± 7.63 | 71.04 ± 8.85 | 71.57 ± 8.87 | 38,852 |
| MAML | 2D | 52.17 ± 2.42 | 58.37 ± 1.42 | 59.59 ± 1.20 | 112,964 |
| Reptile | 1D | 50.90 ± 6.26 | 54.55 ± 5.53 | 55.01 ± 5.79 | 38,852 |
| Reptile | 2D | 54.40 ± 1.92 | 59.71 ± 2.58 | 60.90 ± 2.65 | 112,964 |
| ProtoNet | 1D | 60.29 ± 5.99 | 63.54 ± 7.50 | 64.23 ± 7.55 | 37,824 |
| ProtoNet | 2D | 61.18 ± 0.88 | 68.63 ± 2.58 | 70.04 ± 2.88 | 111,936 |
| RelationNet | 1D | 64.89 ± 1.70 | 70.03 ± 4.19 | 70.16 ± 5.72 | 75,601 |
| RelationNet | 2D | 64.26 ± 0.98 | 70.46 ± 2.79 | 71.57 ± 3.15 | 223,441 |

Mean accuracy ± sample SD across three training seeds. Each seed averages 200 paired target episodes.
