import org.jgrapht.alg.flow.PushRelabelMFImpl;
import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;

import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BinarySearchParallel {

    private final String S_NODE = "s_node";

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph;
    private final Set<String> computeNodes;

    private DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> flowGraph;
    private Map<String, PushRelabelMFImpl<String, DefaultWeightedEdge>> mfMap;

    private static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    public BinarySearchParallel(DefaultDirectedWeightedGraph<String, DefaultWeightedEdge> graph,
                                Collection<String> computeNodes) {
        this.graph = graph;
        this.computeNodes = Set.copyOf(computeNodes);

        this.flowGraph = new DefaultDirectedWeightedGraph<>(DefaultWeightedEdge.class);
        graph.vertexSet().forEach(v -> this.flowGraph.addVertex(v));
        graph.edgeSet().forEach(e -> {
            String u = graph.getEdgeSource(e);
            String v = graph.getEdgeTarget(e);
            DefaultWeightedEdge e2 = this.flowGraph.addEdge(u, v);
            double weight = graph.getEdgeWeight(e);
            assert Math.abs(weight - Math.round(weight)) < 1e-6;
            this.flowGraph.setEdgeWeight(e2, Math.round(weight));
        });

        this.flowGraph.addVertex(S_NODE);
        this.computeNodes.forEach(v -> this.flowGraph.addEdge(S_NODE, v));

        this.mfMap = this.computeNodes.parallelStream()
                .collect(Collectors.toMap(Function.identity(), v -> new PushRelabelMFImpl<>(this.flowGraph)));
    }

    public Pair<Double, Integer> search() {
        int N = this.computeNodes.size();
        int min_bw = this.computeNodes.parallelStream()
                .mapToInt(v -> (int) Math.round(this.graph.incomingEdgesOf(v).stream()
                        .mapToDouble(e -> this.graph.getEdgeWeight(e))
                        .sum()))
                .min().getAsInt();
        double lb = (double) (N - 1) / min_bw;
        double rb = N - 1;
        double end_range = 1.0 / (min_bw * min_bw);
        while (rb - lb > end_range) {
            double mid = (lb + rb) / 2;
            this.computeNodes.parallelStream().forEach(v -> {
                DefaultWeightedEdge e = this.flowGraph.getEdge(S_NODE, v);
                this.flowGraph.setEdgeWeight(e, 1 / mid);
            });
            double fval = this.mfMap.entrySet().parallelStream()
                    .mapToDouble(entry -> entry.getValue().calculateMaximumFlow(S_NODE, entry.getKey()))
                    .min().getAsDouble();
            if (fval < N / mid - 1e-6) {
                lb = mid;
            } else {
                rb = mid;
            }
        }
        double mid = (lb + rb) / 2;
        int denom = IntStream.rangeClosed(1, min_bw).parallel()
                .mapToObj(b -> Pair.of(b, Math.abs(mid * b - Math.round(mid * b))))
                .min(Comparator.comparingDouble(Pair::getSecond))
                .get()
                .getFirst();
        int num = (int) Math.round(mid * denom);

        double U = (double) num / gcd(denom,
                graph.edgeSet().parallelStream()
                        .mapToInt(e -> (int) Math.round(graph.getEdgeWeight(e)))
                        .reduce(BinarySearchParallel::gcd)
                        .getAsInt());
        int k = (int) Math.round(U / mid);

        assert Math.abs((double) num / denom - U / k) < 1e-6;
        assert this.graph.edgeSet().parallelStream().allMatch(e -> {
            double weight = this.graph.getEdgeWeight(e);
            return Math.abs(weight * U - Math.round(weight * U)) < 1e-6;
        });
        return Pair.of(U, k);
    }

    public Pair<Double, Integer> search_fixed_k(int k) {
        int N = this.computeNodes.size();
        int min_bw = this.computeNodes.parallelStream()
                .mapToInt(v -> (int) Math.round(this.graph.incomingEdgesOf(v).stream()
                        .mapToDouble(e -> this.graph.getEdgeWeight(e))
                        .sum()))
                .min().getAsInt();
        int max_b = this.graph.edgeSet().parallelStream()
                .mapToInt(e -> (int) Math.round(this.graph.getEdgeWeight(e)))
                .max()
                .getAsInt();
        double LU = (double) (N - 1) / min_bw * k;

        this.computeNodes.parallelStream().forEach(v -> {
            DefaultWeightedEdge e = this.flowGraph.getEdge(S_NODE, v);
            this.flowGraph.setEdgeWeight(e, k);
        });
        double finalLU = LU;
        this.graph.edgeSet().forEach(e -> {
            double weight = this.graph.getEdgeWeight(e) * finalLU;
            this.flowGraph.setEdgeWeight(
                    this.flowGraph.getEdge(this.graph.getEdgeSource(e), this.graph.getEdgeTarget(e)),
                    Math.abs(weight - Math.round(weight)) < 1e-6 ? Math.round(weight) : Math.floor(weight));
        });
        double fval0 = this.mfMap.entrySet().parallelStream()
                .mapToDouble(entry -> entry.getValue().calculateMaximumFlow(S_NODE, entry.getKey()))
                .min().getAsDouble();

        if (fval0 >= N * k - 1e-6) {
            int denom = IntStream.rangeClosed(1, max_b).parallel()
                    .mapToObj(b -> Pair.of(b, Math.abs(finalLU * b - Math.round(finalLU * b))))
                    .min(Comparator.comparingDouble(Pair::getSecond))
                    .get()
                    .getFirst();
            double U = (double) Math.round(LU * denom) / denom;
            assert Math.abs(LU - U) < 1e-6;
            return Pair.of(U, k);
        } else {
            double RU = (N - 1) * k;
            double end_range = 1.0 / (max_b * max_b);
            while (RU - LU > end_range) {
                double mid = (LU + RU) / 2;
                this.graph.edgeSet().forEach(e -> {
                    double weight = this.graph.getEdgeWeight(e) * mid;
                    this.flowGraph.setEdgeWeight(
                            this.flowGraph.getEdge(this.graph.getEdgeSource(e), this.graph.getEdgeTarget(e)),
                            Math.abs(weight - Math.round(weight)) < 1e-6 ? Math.round(weight) : Math.floor(weight));
                });
                double fval = this.mfMap.entrySet().parallelStream()
                        .mapToDouble(entry -> entry.getValue().calculateMaximumFlow(S_NODE, entry.getKey()))
                        .min().getAsDouble();
                if (fval < N * k - 1e-6) {
                    LU = mid;
                } else {
                    RU = mid;
                }
            }
            double mid = (LU + RU) / 2;
            int denom = IntStream.rangeClosed(1, max_b).parallel()
                    .mapToObj(b -> Pair.of(b, Math.abs(mid * b - Math.round(mid * b))))
                    .min(Comparator.comparingDouble(Pair::getSecond))
                    .get()
                    .getFirst();
            double U = (double) Math.round(LU * denom) / denom;
            return Pair.of(U, k);
        }
    }

}
