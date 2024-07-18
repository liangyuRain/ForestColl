import xml.etree.ElementTree as ET
from xml.dom import minidom
import networkx as nx
from fractions import Fraction
import math

epsilon = 1e-10
max_denom = 10000


def isclose(a, b):
    return math.isclose(a, b, abs_tol=epsilon, rel_tol=0.0)


class OptimalBranchingsAlgo:
    s_node = "SOURCE"

    def __init__(self, topo, capacitated: bool = False, compute_nodes=None):
        self.flow_graph = nx.DiGraph()
        self.flow_graph.add_nodes_from(topo.nodes())

        self.edge_capacity = {}
        for a, b in topo.edges():
            if a == b:
                continue
            if self.flow_graph.has_edge(a, b):
                assert not capacitated, "capacitated edges are not supported in multigraph"
                self.edge_capacity[a, b] += 1
            else:
                self.flow_graph.add_edge(a, b)
                if capacitated:
                    w = topo[a][b]['capacity']
                    assert w > 0 and type(w) is int
                    self.edge_capacity[a, b] = w
                else:
                    self.edge_capacity[a, b] = 1

        if compute_nodes is None:
            compute_nodes = topo.nodes()
        self.compute_nodes = set(compute_nodes)
        self.flow_graph.add_node(OptimalBranchingsAlgo.s_node)
        for n in self.compute_nodes:
            self.flow_graph.add_edge(OptimalBranchingsAlgo.s_node, n)

    def test(self, U: float, k: int, floor: bool) -> bool:
        for b in self.compute_nodes:
            self.flow_graph[OptimalBranchingsAlgo.s_node][b]['capacity'] = k
        for (a, b), count in self.edge_capacity.items():
            capacity = count * U
            if type(capacity) is float and floor:
                if isclose(capacity, round(capacity)):
                    capacity = round(capacity)
                else:
                    capacity = math.floor(capacity)
            self.flow_graph[a][b]['capacity'] = capacity

        for v in self.compute_nodes:
            fval = nx.maximum_flow_value(self.flow_graph, OptimalBranchingsAlgo.s_node, v)
            if fval < len(self.compute_nodes) * k - epsilon:
                return False
        return True

    def binary_search(self) -> tuple[float, int]:
        N = len(self.compute_nodes)
        ingress_bw = {}
        for (_, b), count in self.edge_capacity.items():
            if b not in self.compute_nodes:
                continue
            ingress_bw[b] = ingress_bw.get(b, 0) + count
        min_bw = min(ingress_bw.values())
        assert type(min_bw) is int
        lb = (N - 1) / min_bw
        rb = N - 1
        assert rb >= lb
        end_range = 1 / min_bw ** 2
        while rb - lb > end_range:
            mid = (lb + rb) / 2
            if self.test(1, 1 / mid, False):
                rb = mid
            else:
                lb = mid
        mid = (lb + rb) / 2
        one_div_x_star = Fraction(mid).limit_denominator(min_bw)

        U = one_div_x_star.numerator / math.gcd(*(list(self.edge_capacity.values()) + [one_div_x_star.denominator]))
        k = round(U / one_div_x_star)

        assert isclose(U / k, one_div_x_star)
        for v in self.edge_capacity.values():
            assert isclose(U * v, round(U * v))

        return U, k

    def binary_search_fixed_k(self, k: int) -> tuple[float, int]:
        N = len(self.compute_nodes)
        ingress_bw = {}
        for (_, b), count in self.edge_capacity.items():
            if b not in self.compute_nodes:
                continue
            ingress_bw[b] = ingress_bw.get(b, 0) + count
        min_bw = min(ingress_bw.values())
        max_b = max(self.edge_capacity.values())
        assert type(max_b) is int
        LU = (N - 1) / min_bw * k
        if self.test(LU, k, True):
            U_star = Fraction(LU).limit_denominator(max_b)
            assert isclose(U_star, LU)
        else:
            RU = (N - 1) * k
            while RU - LU > 1 / max_b ** 2:
                U = (LU + RU) / 2
                if self.test(U, k, True):
                    RU = U
                else:
                    LU = U
            U_star = Fraction((LU + RU) / 2).limit_denominator(max_b)
        assert self.test(float(U_star), k, True)
        assert any(isclose(U_star * b, round(U_star * b)) for b in self.edge_capacity.values())
        return float(U_star), k

    # The return value x represents a runtime of M*x. Assuming weights of edges are true bandwidths.
    def convertToRuntime(self, U, k: int) -> float:
        return U / (k * len(self.compute_nodes))

    def removeSwitchNodes(self, U, k: int):
        """
        Returns:
            flow_graph: a graph with switch nodes and source node removed. Edge in the graph corresponds to a path
        (possibly single-edge path) in the original topology such that the paths of any two edges are edge-disjoint. The
        returned graph has the same bandwidth performance as the original topology has.
            edge_map: the map used to map the edges in flow_graph back to paths in original topology. The map is recursive
        if original graph has neighboring edges. The map is guaranteed to be acyclic, i.e., edges of original topology are
        leaves in the recursion tree.
            As for capacity, the edge_map guarantees:
                1. reslt_graph[u][t]['capacity'] = sum(edge_map[u, t][s] for all s)
                2. edge_map[u, t][s] = sum(edge_map[u, s][w] for all w) if (u, s) in edge_map else original_graph[u][s]['capacity']
                                     = sum(edge_map[s, t][w] for all w) if (s, t) in edge_map else original_graph[s][t]['capacity']
        """
        flow_graph = nx.DiGraph(self.flow_graph)
        for b in self.compute_nodes:
            flow_graph[OptimalBranchingsAlgo.s_node][b]['capacity'] = k
        for (a, b), count in self.edge_capacity.items():
            capacity = count * U
            if type(capacity) is float:
                if isclose(capacity, round(capacity)):
                    capacity = round(capacity)
                else:
                    capacity = math.floor(capacity)
            assert type(capacity) is int
            flow_graph[a][b]['capacity'] = capacity

        edge_map = {}
        for w in sorted(w for w in flow_graph.nodes()
                        if w not in self.compute_nodes and w != OptimalBranchingsAlgo.s_node):
            assert w not in self.compute_nodes and w != OptimalBranchingsAlgo.s_node
            for u in sorted(flow_graph.predecessors(w)):
                assert flow_graph[u][w]['capacity'] > 0
                # Sometimes one wants to prioritize splitting edges (u,w),(w,t) such that u,t are not in the same cluster
                # for t in sorted(flow_graph.successors(w), key=lambda t: t if t[0] != u[0] else (1e10, t[1])):
                for t in sorted(flow_graph.successors(w)):
                    assert flow_graph[w][t]['capacity'] > 0
                    M = min(flow_graph[u][w]['capacity'], flow_graph[w][t]['capacity'])
                    flow_graph.add_edge(u, OptimalBranchingsAlgo.s_node)
                    cpct_ut = flow_graph[u][t]['capacity'] if flow_graph.has_edge(u, t) else 0
                    assert cpct_ut > 0 or not flow_graph.has_edge(u, t)
                    if cpct_ut == 0:
                        flow_graph.add_edge(u, t)
                    else:
                        del flow_graph[u][t]['capacity']
                    for v in self.compute_nodes:
                        if u == v or t == v:
                            continue
                        cpct_vw = flow_graph[v][w]['capacity'] if flow_graph.has_edge(v, w) else 0
                        assert cpct_vw > 0 or not flow_graph.has_edge(v, w)
                        if cpct_vw == 0:
                            flow_graph.add_edge(v, w)
                        else:
                            del flow_graph[v][w]['capacity']
                        fval = nx.maximum_flow_value(flow_graph, u, w) - len(self.compute_nodes) * k
                        M = min(fval, M)
                        if cpct_vw == 0:
                            flow_graph.remove_edge(v, w)
                        else:
                            flow_graph[v][w]['capacity'] = cpct_vw
                        if M == 0:
                            break
                    flow_graph.remove_edge(u, OptimalBranchingsAlgo.s_node)
                    if M > 0:
                        flow_graph.add_edge(w, OptimalBranchingsAlgo.s_node)
                        for v in self.compute_nodes:
                            if v != u:
                                cpct_vt = flow_graph[v][t]['capacity'] if flow_graph.has_edge(v, t) else 0
                                assert cpct_vt > 0 or not flow_graph.has_edge(v, t)
                                if cpct_vt == 0:
                                    flow_graph.add_edge(v, t)
                                else:
                                    del flow_graph[v][t]['capacity']
                                fval = nx.maximum_flow_value(flow_graph, w, t) - len(self.compute_nodes) * k
                                if cpct_vt == 0:
                                    flow_graph.remove_edge(v, t)
                                else:
                                    flow_graph[v][t]['capacity'] = cpct_vt
                            else:
                                fval = nx.maximum_flow_value(flow_graph, w, t) - len(self.compute_nodes) * k
                            M = min(fval, M)
                            if M == 0:
                                break
                        flow_graph.remove_edge(w, OptimalBranchingsAlgo.s_node)
                    if cpct_ut == 0:
                        flow_graph.remove_edge(u, t)
                    else:
                        flow_graph[u][t]['capacity'] = cpct_ut
                    if M == 0:
                        continue
                    if (u, t) not in edge_map:
                        edge_map[u, t] = {}
                    if w not in edge_map[u, t]:
                        edge_map[u, t][w] = 0
                    edge_map[u, t][w] += M
                    flow_graph[u][w]['capacity'] -= M
                    flow_graph[w][t]['capacity'] -= M
                    if flow_graph.has_edge(u, t):
                        flow_graph[u][t]['capacity'] += M
                    elif u != t:
                        flow_graph.add_edge(u, t, capacity=M)
                    if flow_graph[w][t]['capacity'] == 0:
                        flow_graph.remove_edge(w, t)
                    if flow_graph[u][w]['capacity'] == 0:
                        break
                assert flow_graph[u][w]['capacity'] == 0
                flow_graph.remove_edge(u, w)
            assert flow_graph.out_degree(w) == flow_graph.in_degree(w) == 0
        for n in list(flow_graph.nodes()):
            if n not in self.compute_nodes and n != OptimalBranchingsAlgo.s_node:
                assert flow_graph.out_degree(n) == flow_graph.in_degree(n) == 0
                flow_graph.remove_node(n)
        for v in self.compute_nodes:
            fval = nx.maximum_flow_value(flow_graph, OptimalBranchingsAlgo.s_node, v)
            assert fval > len(self.compute_nodes) * k - epsilon
        flow_graph.remove_node(OptimalBranchingsAlgo.s_node)
        return flow_graph, edge_map


class BranchingGenerator:

    def __init__(self, topo: nx.DiGraph, k: int):
        for a, b in topo.edges():
            assert 'capacity' in topo[a][b]
        self.N = topo.number_of_nodes()
        self.k = k

        assert OptimalBranchingsAlgo(topo, capacitated=True).test(1, k, False)

        self.flow_graph = nx.DiGraph(topo)
        self.s_nodes = {(u, 0): ('SOURCE', u, 0) for u in topo.nodes()}
        self.Tcount = {u: 1 for u in topo.nodes()}
        self.flow_graph.add_nodes_from(self.s_nodes.values())
        for (u, _), s in self.s_nodes.items():
            self.flow_graph.add_edge(s, u)
        self.num_of_trees = len(self.s_nodes) * k

        self.Ts = {(u, i): [] for u, i in self.s_nodes.keys()}
        self.Vs = {(u, i): {u} for u, i in self.s_nodes.keys()}
        self.Cs = {(u, i): k for u, i in self.s_nodes.keys()}

    def new_tree(self, root, edges, nodes, num) -> tuple:
        index = self.Tcount[root]
        self.Tcount[root] += 1
        s_node = ('SOURCE', root, index)
        self.s_nodes[root, index] = s_node
        self.flow_graph.add_node(s_node)
        for v in nodes:
            self.flow_graph.add_edge(s_node, v)
        self.Ts[root, index] = list(edges)
        self.Vs[root, index] = set(nodes)
        self.Cs[root, index] = num
        return root, index

    def test_edge(self, w, j, x, y):
        c_sum = 0
        for (u, i), s in self.s_nodes.items():
            assert not self.flow_graph.has_edge(x, s)
            if (u, i) != (w, j):
                self.flow_graph.add_edge(x, s, capacity=self.Cs[u, i])
                c_sum += self.Cs[u, i]
        assert self.Cs[w, j] > 0 and self.flow_graph[x][y]['capacity'] > 0
        chunks = min(
            self.Cs[w, j], self.flow_graph[x][y]['capacity'],
            nx.maximum_flow_value(self.flow_graph, x, y) - c_sum
        )
        for (u, i), s in self.s_nodes.items():
            if (u, i) != (w, j):
                self.flow_graph.remove_edge(x, s)
            assert not self.flow_graph.has_edge(x, s)
        return chunks

    # Assume every node in the topology is compute node.
    # Return Ts such that Ts[u, i] is the edges of i-th out-tree rooted at u.
    def generate(self) -> tuple[dict[tuple, list], dict[tuple, int]]:
        trees = sorted(self.s_nodes.keys())
        while len(trees) > 0:
            w, j = trees.pop(0)
            s_node = self.s_nodes[w, j]
            bfs_queue = list(self.Vs[w, j])
            while len(bfs_queue) > 0:
                x = bfs_queue.pop(0)
                assert x in self.Vs[w, j]
                for y in sorted(self.flow_graph.successors(x)):
                    if y in self.Vs[w, j]:
                        continue
                    chunks = self.test_edge(w, j, x, y)
                    if chunks == 0:
                        continue
                    bfs_queue.append(y)
                    if chunks < self.Cs[w, j]:
                        trees.append(self.new_tree(w, self.Ts[w, j], self.Vs[w, j], self.Cs[w, j] - chunks))
                        self.Cs[w, j] = chunks
                    self.Ts[w, j].append((x, y))
                    self.Vs[w, j].add(y)
                    assert not self.flow_graph.has_edge(s_node, y)
                    self.flow_graph.add_edge(s_node, y)
                    self.flow_graph[x][y]['capacity'] -= chunks
                    if self.flow_graph[x][y]['capacity'] == 0:
                        self.flow_graph.remove_edge(x, y)
            assert len(self.Vs[w, j]) == self.N
            assert len(self.Ts[w, j]) == self.N - 1
            self.flow_graph.remove_node(self.s_nodes[w, j])
            del self.s_nodes[w, j]
            self.num_of_trees -= self.Cs[w, j]
        assert self.num_of_trees == 0
        for w, count in self.Tcount.items():
            root_sum = 0
            for j in range(count):
                assert len(self.Vs[w, j]) == self.N
                assert len(self.Ts[w, j]) == self.N - 1
                root_sum += self.Cs[w, j]
            assert root_sum == self.k
        return self.Ts, self.Cs


class SymmtricBranchingGenerator:
    SOURCE = "SOURCE"

    def __init__(self, topo: nx.DiGraph, k: int, syms: list[dict]):
        for a, b in topo.edges():
            assert 'capacity' in topo[a][b]
        self.N = topo.number_of_nodes()
        self.k = k
        self.compute_nodes = list(topo.nodes())

        assert OptimalBranchingsAlgo(topo, capacitated=True).test(1, k, False)

        self.flow_graph = nx.DiGraph(topo)
        self.s_nodes = {(u, 0): ('SOURCE', u, 0) for u in topo.nodes()}
        self.flow_graph.add_nodes_from(self.s_nodes.values())
        for (u, _), s in self.s_nodes.items():
            self.flow_graph.add_edge(SymmtricBranchingGenerator.SOURCE, s, capacity=k)
            self.flow_graph.add_edge(s, u)
        self.num_of_trees = len(self.s_nodes) * k

        self.syms = syms
        remain_nodes = set(topo.nodes())
        self.bases = set()
        for a in sorted(topo.nodes()):
            if a not in remain_nodes:
                continue
            self.bases.add(a)
            for sym in syms:
                b = sym[a]
                assert b in remain_nodes
                remain_nodes.remove(b)
        assert len(remain_nodes) == 0, "Check if identity is included in syms"
        self.bases = list(sorted(self.bases))
        assert topo.number_of_nodes() % len(self.bases) == 0

        self.Tcount = {u: 1 for u in self.bases}
        self.Ts = {(u, 0): [] for u in self.bases}
        self.Vs = {(u, 0): {u} for u in self.bases}
        self.Cs = {(u, 0): k for u in self.bases}

    def new_tree(self, root, edges, nodes, num) -> tuple:
        assert root in self.bases
        index = self.Tcount[root]
        self.Tcount[root] += 1
        self.Ts[root, index] = list(edges)
        self.Vs[root, index] = set(nodes)
        self.Cs[root, index] = num

        for sym in self.syms:
            s_node = ('SOURCE', sym[root], index)
            self.s_nodes[sym[root], index] = s_node
            self.flow_graph.add_node(s_node)
            self.flow_graph.add_edge(SymmtricBranchingGenerator.SOURCE, s_node, capacity=num)
            for v in nodes:
                self.flow_graph.add_edge(s_node, sym[v])

        return root, index

    def test_edge(self, w, j, x, y, nodes):
        for sym in self.syms:
            tmp_s_node = ('temp', self.s_nodes[sym[w], j])
            self.flow_graph.add_node(tmp_s_node)
            self.flow_graph.add_edge(SymmtricBranchingGenerator.SOURCE, tmp_s_node, capacity=0)
            for v in nodes:
                assert sym[v] != sym[y]
                self.flow_graph.add_edge(tmp_s_node, sym[v])
            self.flow_graph.add_edge(tmp_s_node, sym[y])

        chunks = 0
        momentum = 1
        while True:
            chunks += momentum
            if chunks > self.Cs[w, j] or chunks > self.flow_graph[x][y]['capacity']:
                chunks -= momentum
                if momentum == 1:
                    break
                else:
                    momentum = 1
                    continue
            for sym in self.syms:
                s_node = self.s_nodes[sym[w], j]
                assert chunks <= self.flow_graph[sym[x]][sym[y]]['capacity']
                self.flow_graph[sym[x]][sym[y]]['capacity'] -= chunks
                self.flow_graph[SymmtricBranchingGenerator.SOURCE][s_node]['capacity'] -= chunks
                self.flow_graph[SymmtricBranchingGenerator.SOURCE]['temp', s_node]['capacity'] += chunks
            res = True
            for v in self.compute_nodes:
                fval = nx.maximum_flow_value(self.flow_graph, SymmtricBranchingGenerator.SOURCE, v)
                if fval < self.num_of_trees - epsilon:
                    res = False
                    break
            for sym in self.syms:
                s_node = self.s_nodes[sym[w], j]
                self.flow_graph[sym[x]][sym[y]]['capacity'] += chunks
                self.flow_graph[SymmtricBranchingGenerator.SOURCE][s_node]['capacity'] += chunks
                self.flow_graph[SymmtricBranchingGenerator.SOURCE]['temp', s_node]['capacity'] -= chunks
            if res:
                momentum *= 2
            else:
                chunks -= momentum
                if momentum == 1:
                    break
                else:
                    momentum = 1

        for sym in self.syms:
            tmp_s_node = ('temp', self.s_nodes[sym[w], j])
            assert self.flow_graph[SymmtricBranchingGenerator.SOURCE][tmp_s_node]['capacity'] == 0
            self.flow_graph.remove_node(tmp_s_node)
        return chunks

    def generate(self) -> tuple[dict[tuple, list], dict[tuple, int]]:
        trees = sorted(self.Ts.keys())
        while len(trees) > 0:
            w, j = trees.pop(0)
            bfs_queue = list(self.Vs[w, j])
            while len(bfs_queue) > 0:
                x = bfs_queue.pop(0)
                assert x in self.Vs[w, j]
                for y in sorted(self.flow_graph.successors(x)):
                    if y in self.Vs[w, j]:
                        continue
                    chunks = self.test_edge(w, j, x, y, self.Vs[w, j])
                    if chunks == 0:
                        continue
                    bfs_queue.append(y)
                    if chunks < self.Cs[w, j]:
                        trees.append(self.new_tree(w, self.Ts[w, j], self.Vs[w, j], self.Cs[w, j] - chunks))
                        self.Cs[w, j] = chunks
                        for sym in self.syms:
                            s_node = self.s_nodes[sym[w], j]
                            self.flow_graph[SymmtricBranchingGenerator.SOURCE][s_node]['capacity'] = chunks
                    self.Ts[w, j].append((x, y))
                    self.Vs[w, j].add(y)
                    for sym in self.syms:
                        s_node = self.s_nodes[sym[w], j]
                        assert not self.flow_graph.has_edge(s_node, sym[y])
                        self.flow_graph.add_edge(s_node, sym[y])
                        self.flow_graph[sym[x]][sym[y]]['capacity'] -= chunks
                        if self.flow_graph[sym[x]][sym[y]]['capacity'] == 0:
                            self.flow_graph.remove_edge(sym[x], sym[y])
            assert len(self.Vs[w, j]) == self.N
            assert len(self.Ts[w, j]) == self.N - 1
            for sym in self.syms:
                self.flow_graph.remove_node(self.s_nodes[sym[w], j])
                del self.s_nodes[sym[w], j]
                self.num_of_trees -= self.Cs[w, j]
        assert self.num_of_trees == 0
        for w, count in self.Tcount.items():
            root_sum = 0
            for j in range(count):
                assert len(self.Vs[w, j]) == self.N
                assert len(self.Ts[w, j]) == self.N - 1
                root_sum += self.Cs[w, j]
            assert root_sum == self.k

        res_Ts = {}
        res_Cs = {}
        for (w, j), edges in self.Ts.items():
            for sym in self.syms:
                res_Ts[sym[w], j] = [(sym[x], sym[y]) for x, y in edges]
                res_Cs[sym[w], j] = self.Cs[w, j]
        return res_Ts, res_Cs


def _match(ls1, vs1, ls2, vs2, combiner=lambda a, b: a + b):
    """
        Given ls1, vs1 = [l11, ..., l1n], [v11, ..., v1n] and ls2, vs2 = [l21, ..., l2m], [v21, ..., v2m], the algorithm
    generates res_ls, res_vs = [res_l1, ..., res_lx], [res_v1, ..., res_vx]. Each res_l is the combination of some l1i
    and l2j. Suppose res_la, res_lb, res_lc have l1i, then res_va + res_vb + res_vc <= v1i. The same applies to l2j and
    v2j. In particular, sum(res_v1, ..., res_vx) = sum(v11, ..., v1n) = sum(v21, ..., v2m)
    """
    assert len(ls1) == len(vs1)
    assert len(ls2) == len(vs2)
    res_ls = []
    res_vs = []
    i, j = 0, 0
    while i < len(ls1):
        v = min(vs1[i], vs2[j])
        res_ls.append(combiner(ls1[i], ls2[j]))
        res_vs.append(v)
        vs1[i] -= v
        vs2[j] -= v
        if vs1[i] == 0:
            i += 1
        if vs2[j] == 0:
            j += 1
    assert i == len(ls1) and j == len(ls2)
    assert len(res_ls) == len(res_vs)
    return res_ls, res_vs


def flat_edge_map(edge_map: dict, u, t, c):
    """
        The algorithm generates res_paths = [path1, ..., pathn], res_cs = [c1, ..., cn] such that pathi can sustain
    ci amount of flow. In addition, sum(c1, ..., cn) = c.
    """
    if (u, t) not in edge_map:
        return [[(u, t)]], [c]
    res_paths, res_cs = [], []
    for w, cp in sorted(edge_map[u, t].items(), key=lambda e: e[1]):
        if c >= cp:
            del edge_map[u, t][w]
        else:
            edge_map[u, t][w] -= c
            cp = c
        paths1, cs1 = flat_edge_map(edge_map, u, w, cp)
        paths2, cs2 = flat_edge_map(edge_map, w, t, cp)
        paths, cs = _match(paths1, cs1, paths2, cs2)
        assert len(paths) == len(cs)
        res_paths.extend(paths)
        res_cs.extend(cs)
        c -= cp
        assert c >= 0
        if c == 0:
            break
    # c may not be 0 at this point. When edge splitting (u, w), (w, t), it is possible that there is edge (u, t)
    # already existing.
    if c > 0:
        res_paths.append([(u, t)])
        res_cs.append(c)
        assert len(edge_map[u, t]) == 0
        del edge_map[u, t]
    indics = sorted(range(len(res_cs)), reverse=True, key=lambda i: res_cs[i])
    res_paths = [res_paths[i] for i in indics]
    res_cs = [res_cs[i] for i in indics]
    return res_paths, res_cs


# Function makes changes to the parameter edge_map
def mapBackSwitchNodes(input_Ts: dict[tuple, list], input_Cs: dict[tuple, int], edge_map: dict):
    res_Ts = {}
    res_Cs = {}
    Tcounts = {}
    for (u, i), T in input_Ts.items():
        Ts, Cs = [[]], [input_Cs[u, i]]
        for x, y in T:
            Ts2, Cs2 = flat_edge_map(edge_map, x, y, input_Cs[u, i])
            Ts, Cs = _match(Ts, Cs, Ts2, Cs2, combiner=lambda a, b: a + [b])
        assert len(Ts) == len(Cs)
        if u not in Tcounts:
            Tcounts[u] = 0
        for j in range(len(Ts)):
            index = Tcounts[u] + j
            res_Ts[u, index] = Ts[j]
            res_Cs[u, index] = Cs[j]
        Tcounts[u] += len(Ts)
    return res_Ts, res_Cs


def is_arborescence(G: nx.DiGraph):
    # Check if the graph is a directed acyclic graph (DAG)
    if not nx.is_directed_acyclic_graph(G):
        return False

    root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # Check if there's a single root node
    if len(root_nodes) != 1:
        return False

    # Check if all other nodes have an in-degree of exactly 1
    other_nodes = [node for node, in_degree in G.in_degree() if in_degree != 0]
    if not all(G.in_degree(node) == 1 for node in other_nodes):
        return False

    return True


def to_capacitated_graph(topo) -> nx.DiGraph:
    res = nx.DiGraph()
    res.add_nodes_from(topo.nodes())
    for a, b in topo.edges():
        capacity = 1 if 'capacity' not in topo[a][b] else topo[a][b]['capacity']
        if res.has_edge(a, b):
            res[a][b]['capacity'] += capacity
        else:
            res.add_edge(a, b, capacity=capacity)
    return res


def scale_to_integer_capacities(topo) -> tuple[nx.DiGraph, float]:
    fcs = []
    for a, b in topo.edges():
        c = topo[a][b]['capacity']
        fc = Fraction(c).limit_denominator(max_denom)
        assert isclose(c, fc)
        fcs.append(fc)
    scale = math.lcm(*[fc.denominator for fc in fcs]) / math.gcd(*[fc.numerator for fc in fcs])

    res = nx.DiGraph()
    res.add_nodes_from(topo.nodes())
    for a, b in topo.edges():
        res.add_edge(a, b, capacity=round(topo[a][b]['capacity'] * scale))
    return res, scale


def optimal_pipeline_spanning_trees(
        topo, compute_nodes=None, fixed_K: int = None, syms: list[dict] = None
) -> tuple[tuple[float, int], tuple[dict, dict]]:
    topo = to_capacitated_graph(topo)
    topo, scale = scale_to_integer_capacities(topo)
    if compute_nodes is None:
        compute_nodes = set(topo.nodes())
    algo = OptimalBranchingsAlgo(topo, capacitated=True, compute_nodes=compute_nodes)
    if fixed_K is None:
        U, k = algo.binary_search()
    else:
        assert type(fixed_K) is int
        U, k = algo.binary_search_fixed_k(fixed_K)
    flow_graph, edge_map = algo.removeSwitchNodes(U, k)

    if syms is None:
        Ts, Cs = BranchingGenerator(flow_graph, k).generate()
    else:
        Ts, Cs = SymmtricBranchingGenerator(flow_graph, k, syms).generate()
    perf = {}
    max_length = 0
    for u, i in Ts.keys():
        assert len(Ts[u, i]) == flow_graph.number_of_nodes() - 1
        test_G = nx.DiGraph()
        test_G.add_nodes_from(flow_graph.nodes())
        test_G.add_edges_from(Ts[u, i])
        assert is_arborescence(test_G)
        for a, b in Ts[u, i]:
            perf[a, b] = perf.get((a, b), 0) + Cs[u, i]
        lengths = nx.single_source_shortest_path_length(test_G, u)
        max_length = max(max_length, max(lengths.values()))
    for (a, b), congest in perf.items():
        assert congest <= flow_graph[a][b]['capacity']

    Ts, Cs = mapBackSwitchNodes(Ts, Cs, edge_map)
    assert len(Ts) == len(Cs)
    perf = {}
    chunks = {}
    for u, i in Ts.keys():
        chunks[u] = chunks.get(u, 0) + Cs[u, i]
        es = []
        for p in Ts[u, i]:
            es.extend(p)
        test_G = nx.DiGraph()
        test_G.add_nodes_from(topo.nodes())
        test_G.add_edges_from(es)
        for a, b in es:
            perf[a, b] = perf.get((a, b), 0) + Cs[u, i]
        lengths = nx.single_source_shortest_path_length(test_G, u)
        for v in compute_nodes:
            assert v in lengths
    for n in chunks.values():
        assert n == k
    max_congest = 0
    for (a, b), congest in perf.items():
        max_congest = max(max_congest, congest / topo[a][b]['capacity'])
    assert max_congest < U + epsilon

    return (U * scale, k), (Ts, Cs)


def spanning_trees_to_xml(Ts, Cs, k) -> str:
    algo = ET.Element("Algo", attrib={
        "nchunkspernode": str(k)
    })
    for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
        ps = {(p[0][0], p[-1][-1]): [p[0][0]] + [p[j][1] for j in range(len(p))] for p in ps}
        test_G = nx.DiGraph()
        test_G.add_nodes_from(p[0] for p in ps.keys())
        test_G.add_nodes_from(p[1] for p in ps.keys())
        test_G.add_edges_from(ps.keys())
        height = max(nx.single_source_shortest_path_length(test_G, u).values())
        tree = ET.SubElement(algo, "tree", attrib={
            "root": str(u),
            "index": str(i),
            "nchunks": str(Cs[u, i]),
            "height": str(height),
        })
        for a, b in list(nx.bfs_edges(test_G, u)):
            send = ET.SubElement(tree, "send", attrib={
                "src": str(a),
                "dst": str(b),
                "path": ','.join(map(str, ps[a, b])),
            })
    s = ET.tostring(algo, 'utf-8')
    s = minidom.parseString(s)
    s = s.toprettyxml(indent="  ")
    s = '\n'.join(s.split('\n')[1:])
    return s
