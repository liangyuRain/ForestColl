import org.jgrapht.Graph;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Main {

    public static final int PCIE_BW = 25;
    public static final int NVLINK_BW = 300;

    public static Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>, Set<String>>
    a100_topology(int nnode) {
        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph =
                new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        Set<String> computeNodes = new HashSet<>();
        String IB = "IBSwitch";
        graph.addVertex(IB);
        for (int a = 0; a < nnode; a++) {
            String NVSwitch = String.format("NVSwitch_%d", a);
            graph.addVertex(NVSwitch);
            for (int b = 0; b < 8; ++b) {
                String gpu = String.format("gpu_%d_%d", a, b);
                computeNodes.add(gpu);
                graph.addVertex(gpu);

                graph.addEdge(NVSwitch, gpu);
                graph.setEdgeWeight(NVSwitch, gpu, NVLINK_BW);
                graph.addEdge(gpu, NVSwitch);
                graph.setEdgeWeight(gpu, NVSwitch, NVLINK_BW);

                graph.addEdge(IB, gpu);
                graph.setEdgeWeight(IB, gpu, PCIE_BW);
                graph.addEdge(gpu, IB);
                graph.setEdgeWeight(gpu, IB, PCIE_BW);
            }
        }
        graph.edgeSet().forEach(e -> {
            assert graph.getEdgeWeight(e) != Graph.DEFAULT_EDGE_WEIGHT;
        });
        return Pair.of(graph, computeNodes);
    }

    public static final int IB_BW = 16;
    public static final int xGMI_BW = 50;

    public static Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>, Set<String>>
    mi250_topology(int nnode) {
        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph =
                new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        Set<String> computeNodes = new HashSet<>();
        String IB = "IBSwitch";
        graph.addVertex(IB);

        int[][] adj_matrix = {
            {0, 4, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {4, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 4, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0},
            {0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0},
            {2, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 1, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 0},
            {0, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0},
            {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0},
            {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2},
            {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 4},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 4, 0},
        };

        for (int a = 0; a < nnode; a++) {
            for (int u = 0; u < 16; ++u) {
                for (int v = 0; v < 16; ++v) {
                    if (adj_matrix[u][v] > 0) {
                        String u_node = String.format("gpu_%d_%d", a, u);
                        String v_node = String.format("gpu_%d_%d", a, v);
                        if (!graph.containsVertex(u_node)) graph.addVertex(u_node);
                        if (!graph.containsVertex(v_node)) graph.addVertex(v_node);
                        graph.addEdge(u_node, v_node);
                        graph.setEdgeWeight(u_node, v_node, adj_matrix[u][v] * xGMI_BW);
                    }

                }
            }
            for (int b = 0; b < 16; ++b) {
                String gpu = String.format("gpu_%d_%d", a, b);
                computeNodes.add(gpu);
                graph.addVertex(gpu);

                graph.addEdge(IB, gpu);
                graph.setEdgeWeight(IB, gpu, IB_BW);
                graph.addEdge(gpu, IB);
                graph.setEdgeWeight(gpu, IB, IB_BW);
            }
        }
        graph.edgeSet().forEach(e -> {
            assert graph.getEdgeWeight(e) != Graph.DEFAULT_EDGE_WEIGHT;
        });
        return Pair.of(graph, computeNodes);
    }

    public static void main(String[] args) {
        long time_start, time_end;

        StringBuilder log = new StringBuilder();
        for (int nnode = 2; nnode <= 128; nnode *= 2) {
            System.out.printf("nnode = %d\n", nnode);

            // Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>, Set<String>> res = a100_topology(nnode);
            Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>, Set<String>> res = mi250_topology(nnode);
            DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph = res.getFirst();
            Set<String> computeNodes = res.getSecond();

            BinarySearchParallel binarySearchAlgo = new BinarySearchParallel(graph, computeNodes);
            time_start = System.nanoTime();
            Pair<Double, Integer> res2 = binarySearchAlgo.search();
            time_end = System.nanoTime();
            double U = res2.getFirst();
            int k = res2.getSecond();
            double binarySearchTime = (time_end - time_start) / 1e9;
            System.out.printf("U = %f, k = %d\n", U, k);

            assert res2.equals(new BinarySearchParallel(mi250_topology(nnode).getFirst(), computeNodes).search());
            double finalU = U;
            graph.edgeSet().forEach(e -> {
                assert Math.abs(finalU * graph.getEdgeWeight(e) - Math.round(finalU * graph.getEdgeWeight(e))) < 1e-6;
            });

            BinarySearchParallel binarySearchAlgo2 = new BinarySearchParallel(graph, computeNodes);
            time_start = System.nanoTime();
            Pair<Double, Integer> resk1 = binarySearchAlgo2.search_fixed_k(1);
            time_end = System.nanoTime();
            if (Math.abs(k / U - resk1.getSecond() / resk1.getFirst()) < 1e-6) {
                System.out.println("k=1 has the same algbw, resetting U and k");
                U = resk1.getFirst();
                k = resk1.getSecond();
                System.out.printf("U = %f, k = %d\n", U, k);
                binarySearchTime += (time_end - time_start) / 1e9;
            }
            System.out.printf("Binary Search Time: %.3fs\n", binarySearchTime);

            EdgeSplittingParallel edgeSplitAlgo = new EdgeSplittingParallel(graph, computeNodes, k, U);
            time_start = System.nanoTime();
            Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>,
                    Map<Pair<String, String>, Map<String, Integer>>> res3 = edgeSplitAlgo.removeSwitchNodes();
            time_end = System.nanoTime();
            double edgeSplitTime = (time_end - time_start) / 1e9;
            System.out.printf("Edge Splitting Time: %.3fs\n", edgeSplitTime);

            DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> scaledTopo = res3.getFirst();
            Map<Pair<String, String>, Map<String, Integer>> edgeMap = res3.getSecond();
            BranchingGenerator branchAlgo = new BranchingGenerator(scaledTopo, k);
            time_start = System.nanoTime();

            branchAlgo.generate();
            time_end = System.nanoTime();
            double branchingTime = (time_end - time_start) / 1e9;
            System.out.printf("Branching Generation Time: %.3fs\n", branchingTime);
            System.out.printf("Total Time: %.3fs\n", binarySearchTime + edgeSplitTime + branchingTime);
            System.out.println();

            log.append(String.format("nnode = %d\n", nnode));
            log.append(String.format("U = %f, k = %d\n", U, k));
            log.append(String.format("Total Time: %.3fs\n", binarySearchTime + edgeSplitTime + branchingTime));
            log.append(String.format("\tBinary Search Time: %.3fs\n", binarySearchTime));
            log.append(String.format("\tEdge Splitting Time: %.3fs\n", edgeSplitTime));
            log.append(String.format("\tBranching Generation Time: %.3fs\n", branchingTime));
            log.append("\n");

            try (FileWriter fw = new FileWriter("log.txt")) {
                fw.write(log.toString());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}