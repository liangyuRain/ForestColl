import org.jgrapht.alg.flow.PushRelabelMFImpl;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class EdgeSplittingParallel {

    private final String S_NODE = "s_node";
    private final int INF_CAPACITY = Integer.MAX_VALUE;

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph;
    private final Set<String> computeNodes;
    private final int k;

    private Map<String, DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> flowGraphCopies;
    private Map<String, PushRelabelMFImpl<String, DefaultWeightedEdge>> mfMap;

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

    public EdgeSplittingParallel(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph,
                                 Collection<String> computeNodes,
                                 int k, double U) {
        graph.vertexSet().forEach(v -> {
            int total_ingress = graph.incomingEdgesOf(v).stream()
                    .map(e -> getWeight(graph, graph.getEdgeSource(e), v)).mapToInt(Math::round).sum();
            int total_egress = graph.outgoingEdgesOf(v).stream()
                    .map(e -> getWeight(graph, v, graph.getEdgeTarget(e))).mapToInt(Math::round).sum();
            assert total_ingress == total_egress;
        });

        this.computeNodes = Set.copyOf(computeNodes);
        this.graph = graph;
        this.k = k;

        this.flowGraphCopies = this.computeNodes.parallelStream()
                .map(v -> {
                    DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> copy =
                            new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
                    graph.vertexSet().forEach(copy::addVertex);
                    graph.edgeSet().forEach(e -> copy.addEdge(graph.getEdgeSource(e), graph.getEdgeTarget(e)));
                    copy.edgeSet().forEach(e -> {
                        double weight = U * getWeight(graph, copy.getEdgeSource(e), copy.getEdgeTarget(e));
                        copy.setEdgeWeight(e,
                                Math.abs(U - Math.round(U)) < 1e-6 ? Math.round(weight) : Math.floor(weight));
                    });

                    copy.addVertex(S_NODE);
                    this.computeNodes.forEach(u -> {
                        DefaultWeightedEdge e = copy.addEdge(S_NODE, u);
                        copy.setEdgeWeight(e, k);
                    });
                    assert this.computeNodes.size() * k == this.computeNodes.stream()
                            .map(u -> (int) Math.round(new PushRelabelMFImpl<>(copy)
                                    .calculateMaximumFlow(S_NODE, u))).mapToInt(Integer::intValue).min().getAsInt();
                    return Pair.of(v, copy);
                }).collect(Collectors.toMap(Pair::getFirst, Pair::getSecond));
        this.mfMap = this.flowGraphCopies.entrySet().parallelStream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> new PushRelabelMFImpl<>(entry.getValue())));
    }

    private int computeFlow(Map.Entry<String, DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>> entry,
                            String u, String w, String t, AtomicBoolean abort, int cpct_ut) {
        if (abort.get()) return 0;

        String v = entry.getKey();
        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> copy = entry.getValue();
        PushRelabelMFImpl<String, DefaultWeightedEdge> mf = this.mfMap.get(v);
        int minfval = Integer.MAX_VALUE;
        int fval;

        if (cpct_ut == 0) copy.addEdge(u, t);
        copy.setEdgeWeight(u, t, INF_CAPACITY);

        copy.addEdge(u, S_NODE);
        copy.setEdgeWeight(u, S_NODE, INF_CAPACITY);

        if (!v.equals(u) && !v.equals(t)) {
            int cpct_vw = copy.containsEdge(v, w) ? getWeight(copy, v, w) : 0;
            assert cpct_vw > 0 || !copy.containsEdge(v, w);
            if (cpct_vw == 0) copy.addEdge(v, w);
            copy.setEdgeWeight(v, w, INF_CAPACITY);
            fval = (int) Math.round(mf.calculateMaximumFlow(u, w)) - this.computeNodes.size() * k;
            assert fval >= 0;
            minfval = Math.min(minfval, fval);
            if (minfval == 0) abort.set(true);
            if (cpct_vw == 0) copy.removeEdge(v, w);
            else copy.setEdgeWeight(v, w, cpct_vw);
        }

        copy.removeEdge(u, S_NODE);

        if (!abort.get()) {
            copy.addEdge(w, S_NODE);
            copy.setEdgeWeight(w, S_NODE, INF_CAPACITY);

            if (!v.equals(u)) {
                int cpct_vt = copy.containsEdge(v, t) ? getWeight(copy, v, t) : 0;
                assert cpct_vt > 0 || !copy.containsEdge(v, t);
                if (cpct_vt == 0) copy.addEdge(v, t);
                copy.setEdgeWeight(v, t, INF_CAPACITY);
                fval = (int) Math.round(mf.calculateMaximumFlow(w, t)) - this.computeNodes.size() * k;
                assert fval >= 0;
                if (cpct_vt == 0) copy.removeEdge(v, t);
                else copy.setEdgeWeight(v, t, cpct_vt);
            } else {
                fval = (int) Math.round(mf.calculateMaximumFlow(w, t)) - this.computeNodes.size() * k;
                assert fval >= 0;
            }
            minfval = Math.min(minfval, fval);
            if (minfval == 0) abort.set(true);

            copy.removeEdge(w, S_NODE);
        } else {
            minfval = 0;
        }

        if (cpct_ut == 0) copy.removeEdge(u, t);
        else copy.setEdgeWeight(u, t, cpct_ut);

        return minfval;
    }

    private static void splitWeight(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> copy,
                                    String u, String w, String t, int M) {
        setWeight(copy, u, w, getWeight(copy, u, w) - M);
        setWeight(copy, w, t, getWeight(copy, w, t) - M);
        if (copy.containsEdge(u, t)) {
            setWeight(copy, u, t, getWeight(copy, u, t) + M);
        } else if (!u.equals(t)) {
            copy.addEdge(u, t);
            setWeight(copy, u, t, M);
        }
        if (getWeight(copy, w, t) == 0) copy.removeEdge(w, t);
    }

    public Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>,
            Map<Pair<String, String>, Map<String, Integer>>> removeSwitchNodes() {
        DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> oneCopy =
                this.flowGraphCopies.values().iterator().next();
        Map<Pair<String, String>, Map<String, Integer>> edge_map = new HashMap<>();
        List<String> switchNodes = this.graph.vertexSet().parallelStream()
                .filter(w -> !this.computeNodes.contains(w))
                .sorted()
                .toList();
        int totalSwitchCapa = switchNodes.parallelStream()
                .mapToInt(w -> oneCopy.incomingEdgesOf(w).stream()
                        .mapToInt(e -> getWeight(oneCopy, oneCopy.getEdgeSource(e), w)).sum())
                .sum();
        int removedSwitchCapa = 0;
        for (String w : switchNodes) {
            for (String u : oneCopy.incomingEdgesOf(w)
                    .stream().map(oneCopy::getEdgeSource).toList()) {
                assert getWeight(oneCopy, u, w) > 0;
                for (String t : oneCopy.outgoingEdgesOf(w)
                        .stream().map(oneCopy::getEdgeTarget).toList()) {
                    assert getWeight(oneCopy, w, t) > 0;

                    int cpct_ut = oneCopy.containsEdge(u, t) ? getWeight(oneCopy, u, t) : 0;
                    assert cpct_ut > 0 || !oneCopy.containsEdge(u, t);

                    AtomicBoolean hasZero = new AtomicBoolean(false);

                    int minfval = this.flowGraphCopies.entrySet().parallelStream()
                            .mapToInt(entry -> computeFlow(entry, u, w, t, hasZero, cpct_ut))
                            .takeWhile(fval -> fval > 0)
                            .min().orElse(0);
                    final int M = Math.min(
                            Math.min(getWeight(oneCopy, u, w), getWeight(oneCopy, w, t)),
                            hasZero.get() ? 0 : minfval
                    );
                    if (M == 0) continue;

                    if (!edge_map.containsKey(Pair.of(u, t))) edge_map.put(Pair.of(u, t), new HashMap<>());
                    if (!edge_map.get(Pair.of(u, t)).containsKey(w)) edge_map.get(Pair.of(u, t)).put(w, 0);
                    edge_map.get(Pair.of(u, t)).put(w, edge_map.get(Pair.of(u, t)).get(w) + M);

                    this.flowGraphCopies.values().parallelStream().forEach(copy -> splitWeight(copy, u, w, t, M));

                    removedSwitchCapa += M;
                    System.out.printf("Removed Switch Capacity: %d / %d\r", removedSwitchCapa, totalSwitchCapa);

                    if (getWeight(oneCopy, u, w) == 0) break;
                }
                this.flowGraphCopies.values().parallelStream().forEach(copy -> {
                    assert getWeight(copy, u, w) == 0;
                    copy.removeEdge(u, w);
                });
            }
            assert this.flowGraphCopies.values().parallelStream().allMatch(copy ->
                    copy.inDegreeOf(w) == 0 && copy.outDegreeOf(w) == 0
            );
        }
        this.flowGraphCopies.values().parallelStream().forEach(copy -> {
            for (String n : copy.vertexSet().stream().toList()) {
                if (!this.computeNodes.contains(n) && !n.equals(S_NODE)) {
                    assert copy.inDegreeOf(n) == 0 && copy.outDegreeOf(n) == 0;
                    copy.removeVertex(n);
                }
            }
            int M = Integer.MAX_VALUE;
            for (String v : this.computeNodes) {
                PushRelabelMFImpl<String, DefaultWeightedEdge> mf = new PushRelabelMFImpl<>(copy);
                M = Math.min(M, (int) Math.round(mf.calculateMaximumFlow(S_NODE, v)));
            }
            assert M == this.computeNodes.size() * k;
        });
        oneCopy.removeVertex(S_NODE);
        return Pair.of(oneCopy, edge_map);
    }

}
