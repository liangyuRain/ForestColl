import org.jgrapht.alg.flow.PushRelabelMFImpl;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BranchingGenerator {

    private static final int NUM_ACCEPT_WORKERS = 4;
    private static final int NUM_REJECT_WORKERS = 4;

    private final int N;
    private final int k;

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> flowGraph;
    private Map<Pair<String, Integer>, String> s_nodes;
    private Map<String, Integer> Tcount;
    private int num_of_trees;

    private Map<Pair<String, Integer>, List<Pair<String, String>>> Ts;
    private Map<Pair<String, Integer>, Set<String>> Vs;
    private Map<Pair<String, Integer>, Integer> Cs;

    private static int getWeight(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> G, String u, String v) {
        double weight = G.getEdgeWeight(G.getEdge(u, v));
        assert Math.abs(weight - Math.round(weight)) < 1e-6;
        return (int) Math.round(weight);
    }

    private static void setWeight(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> G,
                                  String u, String v, int weight) {
        assert weight >= 0;
        G.setEdgeWeight(G.getEdge(u, v), weight);
    }

    private boolean accept_edge(Pair<String, String> xy, int chunks, Pair<String, Integer> wj,
                                List<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> allGraphs,
                                String s_node, Queue<Pair<String, Integer>> trees,
                                ArrayDeque<Pair<String, String>> bfs_queue) {
        boolean split = false;
        String x = xy.getFirst();
        String y = xy.getSecond();
        this.flowGraph.outgoingEdgesOf(y).stream()
                .map(e -> Pair.of(y, this.flowGraph.getEdgeTarget(e)))
                .forEach(bfs_queue::add);
        if (chunks < this.Cs.get(wj)) {
            Pair<String, Integer> wj2 = new_tree(
                    allGraphs, wj.getFirst(), this.Ts.get(wj),
                    this.Vs.get(wj), this.Cs.get(wj) - chunks);
            trees.add(wj2);
            this.Cs.put(wj, chunks);
            split = true;
        }
        this.Ts.get(wj).add(xy);
        this.Vs.get(wj).add(y);

        allGraphs.forEach(graph -> {
            assert !graph.containsEdge(s_node, y);
            graph.addEdge(s_node, y);
            setWeight(graph, s_node, y, Integer.MAX_VALUE);
            setWeight(graph, x, y, getWeight(graph, x, y) - chunks);
            if (getWeight(graph, x, y) == 0) {
                graph.removeEdge(x, y);
            }
        });
        return split;
    }

    public BranchingGenerator(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> scaledTopo, int k) {
        this.N = scaledTopo.vertexSet().size();
        this.k = k;

        assert new BinarySearchParallel(scaledTopo, scaledTopo.vertexSet()).search().equals(Pair.of(1.0, k));

        this.flowGraph = new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        scaledTopo.vertexSet().forEach(v -> this.flowGraph.addVertex(v));
        scaledTopo.edgeSet().forEach(e -> {
            String u = scaledTopo.getEdgeSource(e);
            String v = scaledTopo.getEdgeTarget(e);
            DefaultWeightedEdge e2 = this.flowGraph.addEdge(u, v);
            double weight = scaledTopo.getEdgeWeight(e);
            assert Math.abs(weight - Math.round(weight)) < 1e-6;
            this.flowGraph.setEdgeWeight(e2, Math.round(weight));
        });
        this.s_nodes = scaledTopo.vertexSet().stream()
                .collect(Collectors.toMap(u -> Pair.of(u, 0), u -> String.format("SOURCE_%s_0", u)));
        this.Tcount = scaledTopo.vertexSet().stream()
                .collect(Collectors.toMap(Function.identity(), u -> 1));
        this.s_nodes.values().forEach(v -> this.flowGraph.addVertex(v));
        this.s_nodes.entrySet().forEach(entry -> {
            String s = entry.getValue();
            String u = entry.getKey().getFirst();
            this.flowGraph.addEdge(s, u);
            setWeight(this.flowGraph, s, u, Integer.MAX_VALUE);
        });
        this.num_of_trees = this.s_nodes.size() * k;

        this.Ts = this.s_nodes.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> new ArrayList<>()));
        this.Vs = this.s_nodes.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> new HashSet<>(List.of(entry.getKey().getFirst()))));
        this.Cs = this.s_nodes.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> k));
    }

    private Pair<String, Integer> new_tree(
            List<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> graphs,
            String root, Collection<Pair<String, String>> edges,
            Collection<String> nodes, int num) {
        int index = this.Tcount.get(root);
        this.Tcount.put(root, index + 1);
        String s_node = String.format("SOURCE_%s_%d", root, index);
        this.s_nodes.put(Pair.of(root, index), s_node);

        graphs.parallelStream().forEach(graph -> {
            graph.addVertex(s_node);
            nodes.forEach(v -> {
                graph.addEdge(s_node, v);
                setWeight(graph, s_node, v, Integer.MAX_VALUE);
            });
        });

        Pair<String, Integer> key = Pair.of(root, index);
        this.Ts.put(key, new ArrayList<>(edges));
        this.Vs.put(key, new HashSet<>(nodes));
        this.Cs.put(key, num);
        return key;
    }

    private static int test_edge(
            DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> flowGraph,
            Map<Pair<String, Integer>, String> s_nodes,
            Map<Pair<String, Integer>, Integer> Cs,
            Pair<String, Integer> wj, String x, String y) {
        int c_sum = s_nodes.entrySet().stream()
                .mapToInt(entry -> {
                    String s = entry.getValue();
                    assert !flowGraph.containsEdge(x, s);
                    if (!entry.getKey().equals(wj)) {
                        int C = Cs.get(entry.getKey());
                        flowGraph.addEdge(x, s);
                        setWeight(flowGraph, x, s, C);
                        return C;
                    }
                    return 0;
                })
                .sum();
        assert Cs.get(wj) > 0 && getWeight(flowGraph, x, y) > 0;
        flowGraph.edgeSet().forEach(e -> {
            String u = flowGraph.getEdgeSource(e);
            String v = flowGraph.getEdgeTarget(e);
            assert u != null && v != null;
        });
        int chunks = Math.min(
                Math.min(Cs.get(wj), getWeight(flowGraph, x, y)),
                // The vertex set of this.flowGraph is changed. Cannot reuse MaximumFlow Implementation.
                (int) Math.round(new PushRelabelMFImpl<>(flowGraph).calculateMaximumFlow(x, y) - c_sum)
        );
        s_nodes.entrySet().stream()
                .forEach(entry -> {
                    String s = entry.getValue();
                    if (!entry.getKey().equals(wj)) {
                        flowGraph.removeEdge(x, s);
                    }
                    assert !flowGraph.containsEdge(x, s);
                });
        return chunks;
    }

    public Pair<Map<Pair<String, Integer>, List<Pair<String, String>>>, Map<Pair<String, Integer>, Integer>>
    generate() {
        Queue<Pair<String, Integer>> trees = this.s_nodes.keySet().parallelStream()
                .sorted(Comparator.<Pair<String, Integer>, String>comparing(Pair::getFirst)
                        .thenComparingInt(Pair::getSecond))
                .collect(Collectors.toCollection(ArrayDeque::new));

        List<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> workerFlowGraphs = new ArrayList<>();
        for (int i = 0; i < NUM_ACCEPT_WORKERS + NUM_REJECT_WORKERS; ++i) {
            DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph = new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
            this.flowGraph.vertexSet().forEach(graph::addVertex);
            this.flowGraph.edgeSet().forEach(e -> {
                String u = this.flowGraph.getEdgeSource(e);
                String v = this.flowGraph.getEdgeTarget(e);
                DefaultWeightedEdge e2 = graph.addEdge(u, v);
                double weight = this.flowGraph.getEdgeWeight(e);
                assert Math.abs(weight - Math.round(weight)) < 1e-6;
                graph.setEdgeWeight(e2, Math.round(weight));
            });
            workerFlowGraphs.add(graph);
        }

        List<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> allGraphs =
                Stream.concat(workerFlowGraphs.stream(), Stream.of(this.flowGraph))
                        .toList();

        while (!trees.isEmpty()) {
            System.out.printf("Constructed Trees: %d / %d\r", this.N * this.k - this.num_of_trees, this.N * this.k);
            Pair<String, Integer> wj = trees.poll();
            String s_node = this.s_nodes.get(wj);
            ArrayDeque<Pair<String, String>> bfs_queue = this.Vs.get(wj).stream()
                    .flatMap(x -> this.flowGraph.outgoingEdgesOf(x).stream()
                            .map(e -> this.flowGraph.getEdgeTarget(e))
                            .filter(y -> !this.Vs.get(wj).contains(y))
                            .map(y -> Pair.of(x, y)))
                    .collect(Collectors.toCollection(ArrayDeque::new));
            Set<String> round_visited = new HashSet<>();
            while (!bfs_queue.isEmpty()) {
                round_visited.clear();
                List<Pair<String, String>> accept_guesses = new ArrayList<>();
                List<Pair<String, String>> reject_guesses = new ArrayList<>();
                boolean first_xy = true;
                boolean stop_accept = false;
                for (Pair<String, String> xy : bfs_queue) {
                    String x = xy.getFirst();
                    String y = xy.getSecond();
                    if (this.Vs.get(wj).contains(y)) continue;
                    // The first xy is in accept_guesses.
                    if (!first_xy && reject_guesses.size() < NUM_REJECT_WORKERS) {
                        reject_guesses.add(xy);
                    }
                    first_xy = false;
                    if (!stop_accept && round_visited.add(y)) {
                        accept_guesses.add(xy);
                        if (accept_guesses.size() == NUM_ACCEPT_WORKERS || getWeight(this.flowGraph, x, y) - this.Cs.get(wj) < 0) {
                            stop_accept = true;
                        }
                    }
                    if (reject_guesses.size() == NUM_REJECT_WORKERS && stop_accept) break;
                }
                final int accept_guesses_size = accept_guesses.size();
                final int reject_guesses_size = reject_guesses.size();

                int[] results = IntStream.range(0, accept_guesses_size + reject_guesses_size).parallel().map(j -> {
                    DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph = workerFlowGraphs.get(j);
                    boolean is_accept = j < accept_guesses_size;
                    if (is_accept) {
                        for (int k = 0; k < j; ++k) {
                            Pair<String, String> xy = accept_guesses.get(k);
                            String x = xy.getFirst();
                            String y = xy.getSecond();
                            graph.addEdge(s_node, y);
                            setWeight(graph, s_node, y, Integer.MAX_VALUE);
                            int weight = getWeight(graph, x, y) - this.Cs.get(wj);
                            if (weight > 0) {
                                setWeight(graph, x, y, weight);
                            } else {
                                graph.removeEdge(x, y);
                            }
                        }
                    }

                    Pair<String, String> target_xy = is_accept ? accept_guesses.get(j) : reject_guesses.get(j - accept_guesses_size);
                    String target_x = target_xy.getFirst();
                    String target_y = target_xy.getSecond();
                    int chunks = test_edge(graph, this.s_nodes, this.Cs, wj, target_x, target_y);

                    if (is_accept) {
                        for (int k = j - 1; k >= 0; --k) {
                            Pair<String, String> xy = accept_guesses.get(k);
                            String x = xy.getFirst();
                            String y = xy.getSecond();
                            if (!graph.containsEdge(x, y)) {
                                graph.addEdge(x, y);
                            }
                            setWeight(graph, x, y, getWeight(this.flowGraph, x, y));
                            graph.removeEdge(s_node, y);
                        }
                    }

                    return chunks;
                }).toArray();

                // Even if the first xy is not accepted, it needs to be removed.
                Pair<String, String> last_xy = accept_guesses.getFirst();
                if (results[0] > 0) {
                    for (int j = 0; j < accept_guesses_size; ++j) {
                        Pair<String, String> xy = accept_guesses.get(j);
                        last_xy = xy;

                        int chunks = results[j];
                        if (chunks <= 0) break;

                        boolean split = accept_edge(xy, chunks, wj, allGraphs, s_node, trees, bfs_queue);
                        if (split) break;
                    }
                } else {
                    for (int j = 0; j < reject_guesses_size; ++j) {
                        Pair<String, String> xy = reject_guesses.get(j);
                        last_xy = xy;

                        int chunks = results[j + accept_guesses_size];
                        if (chunks <= 0) continue;

                        accept_edge(xy, chunks, wj, allGraphs, s_node, trees, bfs_queue);
                        break;
                    }
                }
                while (!bfs_queue.remove().equals(last_xy)) ;
                bfs_queue.removeIf(xy -> this.Vs.get(wj).contains(xy.getSecond()));
            }
            assert this.Vs.get(wj).size() == this.N;
            assert this.Ts.get(wj).size() == this.N - 1;
            allGraphs.forEach(graph -> graph.removeVertex(s_node));
            this.s_nodes.remove(wj);
            this.num_of_trees -= this.Cs.get(wj);
        }
        assert this.num_of_trees == 0;
        this.Tcount.entrySet().forEach(entry -> {
            String w = entry.getKey();
            int count = entry.getValue();
            int root_sum = 0;
            for (int j = 0; j < count; ++j) {
                Pair<String, Integer> wj = Pair.of(w, j);
                assert this.Vs.get(wj).size() == this.N;
                assert this.Ts.get(wj).size() == this.N - 1;
                root_sum += this.Cs.get(wj);
            }
            assert root_sum == this.k;
        });
        return Pair.of(this.Ts, this.Cs);
    }

}
