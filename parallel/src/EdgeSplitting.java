import org.jgrapht.alg.flow.PushRelabelMFImpl;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.*;

import java.util.*;

public class EdgeSplitting {

    private final String S_NODE = "s_node";
    private final int INF_CAPACITY = Integer.MAX_VALUE;

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph;
    private final Set<String> computeNodes;
    private final int k;

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> flowGraph;

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

    public EdgeSplitting(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph,
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

        this.flowGraph = new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        graph.vertexSet().forEach(v -> this.flowGraph.addVertex(v));
        graph.edgeSet().forEach(e -> this.flowGraph.addEdge(graph.getEdgeSource(e), graph.getEdgeTarget(e)));
        this.flowGraph.edgeSet().forEach(e -> {
            this.flowGraph.setEdgeWeight(e, Math.round(U * getWeight(graph,
                    this.flowGraph.getEdgeSource(e), this.flowGraph.getEdgeTarget(e))));
        });

        this.flowGraph.addVertex(S_NODE);
        this.computeNodes.forEach(v -> {
            DefaultWeightedEdge e = this.flowGraph.addEdge(S_NODE, v);
            this.flowGraph.setEdgeWeight(e, k);
        });
        assert this.computeNodes.size() * k == this.computeNodes.stream()
                .map(v -> (int) Math.round(new PushRelabelMFImpl<>(this.flowGraph)
                        .calculateMaximumFlow(S_NODE, v))).mapToInt(Integer::intValue).min().getAsInt();
    }

    public Pair<DefaultDirectedWeightedGraph<String, DefaultWeightedEdge>,
            Map<Pair<String, String>, Map<String, Integer>>> removeSwitchNodes() {
        Map<Pair<String, String>, Map<String, Integer>> edge_map = new HashMap<>();
        for (String w : this.graph.vertexSet().stream().sorted().toList()) {
            if (this.computeNodes.contains(w)) continue;
            for (String u : this.flowGraph.incomingEdgesOf(w)
                    .stream().map(e -> this.flowGraph.getEdgeSource(e)).sorted().toList()) {
                assert getWeight(this.flowGraph, u, w) > 0;
                for (String t : this.flowGraph.outgoingEdgesOf(w)
                        .stream().map(e -> this.flowGraph.getEdgeTarget(e)).sorted().toList()) {
                    assert getWeight(this.flowGraph, w, t) > 0;
                    int M = Math.min(getWeight(this.flowGraph, u, w), getWeight(this.flowGraph, w, t));
                    this.flowGraph.addEdge(u, S_NODE);
                    this.flowGraph.setEdgeWeight(u, S_NODE, INF_CAPACITY);

                    int cpct_ut = this.flowGraph.containsEdge(u, t) ? getWeight(this.flowGraph, u, t) : 0;
                    assert cpct_ut > 0 || !this.flowGraph.containsEdge(u, t);
                    if (cpct_ut == 0) this.flowGraph.addEdge(u, t);
                    this.flowGraph.setEdgeWeight(u, t, INF_CAPACITY);

                    for (String v : this.computeNodes) {
                        if (v.equals(u) || v.equals(t)) continue;
                        int cpct_vw = this.flowGraph.containsEdge(v, w) ? getWeight(this.flowGraph, v, w) : 0;
                        assert cpct_vw > 0 || !this.flowGraph.containsEdge(v, w);
                        if (cpct_vw == 0) this.flowGraph.addEdge(v, w);
                        this.flowGraph.setEdgeWeight(v, w, INF_CAPACITY);
                        PushRelabelMFImpl<String, DefaultWeightedEdge> mf = new PushRelabelMFImpl<>(this.flowGraph);
                        int fval = (int) Math.round(mf.calculateMaximumFlow(u, w)) - this.computeNodes.size() * k;
                        assert fval >= 0;
                        M = Math.min(M, fval);
                        if (cpct_vw == 0) this.flowGraph.removeEdge(v, w);
                        else this.flowGraph.setEdgeWeight(v, w, cpct_vw);
                        if (M == 0) break;
                    }
                    this.flowGraph.removeEdge(u, S_NODE);
                    if (M > 0) {
                        this.flowGraph.addEdge(w, S_NODE);
                        this.flowGraph.setEdgeWeight(w, S_NODE, INF_CAPACITY);
                        for (String v : this.computeNodes) {
                            int fval;
                            if (!v.equals(u)) {
                                int cpct_vt = this.flowGraph.containsEdge(v, t) ? getWeight(this.flowGraph, v, t) : 0;
                                assert cpct_vt > 0 || !this.flowGraph.containsEdge(v, t);
                                if (cpct_vt == 0) this.flowGraph.addEdge(v, t);
                                this.flowGraph.setEdgeWeight(v, t, INF_CAPACITY);
                                PushRelabelMFImpl<String, DefaultWeightedEdge> mf = new PushRelabelMFImpl<>(this.flowGraph);
                                fval = (int) Math.round(mf.calculateMaximumFlow(w, t)) - this.computeNodes.size() * k;
                                assert fval >= 0;
                                if (cpct_vt == 0) this.flowGraph.removeEdge(v, t);
                                else this.flowGraph.setEdgeWeight(v, t, cpct_vt);
                            } else {
                                PushRelabelMFImpl<String, DefaultWeightedEdge> mf = new PushRelabelMFImpl<>(this.flowGraph);
                                fval = (int) Math.round(mf.calculateMaximumFlow(w, t)) - this.computeNodes.size() * k;
                                assert fval >= 0;
                            }
                            M = Math.min(M, fval);
                            if (M == 0) break;
                        }
                        this.flowGraph.removeEdge(w, S_NODE);
                    }
                    if (cpct_ut == 0) this.flowGraph.removeEdge(u, t);
                    else this.flowGraph.setEdgeWeight(u, t, cpct_ut);
                    if (M == 0) continue;
                    if (!edge_map.containsKey(Pair.of(u, t))) edge_map.put(Pair.of(u, t), new HashMap<>());
                    if (!edge_map.get(Pair.of(u, t)).containsKey(w)) edge_map.get(Pair.of(u, t)).put(w, 0);
                    edge_map.get(Pair.of(u, t)).put(w, edge_map.get(Pair.of(u, t)).get(w) + M);
                    setWeight(this.flowGraph, u, w, getWeight(this.flowGraph, u, w) - M);
                    setWeight(this.flowGraph, w, t, getWeight(this.flowGraph, w, t) - M);
                    if (this.flowGraph.containsEdge(u, t)) {
                        setWeight(this.flowGraph, u, t, getWeight(this.flowGraph, u, t) + M);
                    } else if (!u.equals(t)) {
                        this.flowGraph.addEdge(u, t);
                        setWeight(this.flowGraph, u, t, M);
                    }
                    if (getWeight(this.flowGraph, w, t) == 0) this.flowGraph.removeEdge(w, t);
                    if (getWeight(this.flowGraph, u, w) == 0) break;
                }
                assert getWeight(this.flowGraph, u, w) == 0;
                this.flowGraph.removeEdge(u, w);
            }
            assert this.flowGraph.inDegreeOf(w) == 0 && this.flowGraph.outDegreeOf(w) == 0;
        }
        for (String n : this.flowGraph.vertexSet().stream().toList()) {
            if (!this.computeNodes.contains(n) && !n.equals(S_NODE)) {
                assert this.flowGraph.inDegreeOf(n) == 0 && this.flowGraph.outDegreeOf(n) == 0;
                this.flowGraph.removeVertex(n);
            }
        }
        int M = Integer.MAX_VALUE;
        for (String v : this.computeNodes) {
            PushRelabelMFImpl<String, DefaultWeightedEdge> mf = new PushRelabelMFImpl<>(this.flowGraph);
            M = Math.min(M, (int) Math.round(mf.calculateMaximumFlow(S_NODE, v)));
        }
        assert M == this.computeNodes.size() * k;
        this.flowGraph.removeVertex(S_NODE);
        return Pair.of(this.flowGraph, edge_map);
    }

}
