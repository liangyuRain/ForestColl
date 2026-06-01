# Parallel ForestColl

A parallel version of ForestColl for constructing **bandwidth-optimal collective-communication
schedules** on GPU network topologies. Given the interconnect topology of a GPU
cluster, it computes the optimal achievable bandwidth and produces a set of
spanning trees (branchings) that realize it.

The pipeline has three stages:

1. **Binary search** for the optimal scaling factor `U` and chunk count `k` that
   maximize achievable bandwidth, using parallel max-flow computations.
2. **Edge splitting** to remove non-compute (switch) nodes from the topology
   while preserving flow, yielding a scaled topology over compute nodes only.
3. **Branching generation** to decompose the scaled topology into the spanning
   trees that define the actual communication schedule.

## Repository layout

| File | Description |
| --- | --- |
| `src/Main.java` | Entry point. Builds A100 / MI250 topologies and runs the full pipeline, timing each stage. |
| `src/BinarySearchParallel.java` | Parallel binary search for optimal `U` and `k` via max-flow. |
| `src/EdgeSplittingParallel.java` | Parallel edge-splitting that removes switch nodes from the topology. |
| `src/EdgeSplitting.java` | Single-threaded reference implementation of edge splitting. |
| `src/BranchingGenerator.java` | Generates the spanning trees (branchings) from the scaled topology. |

## Dependencies

- **Java 17+** (the code uses `List.toList()`, `getFirst()`, and other recent APIs).
- **[JGraphT](https://jgrapht.org/)** core (`org.jgrapht:jgrapht-core`) on the classpath.
