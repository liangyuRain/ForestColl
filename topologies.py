import networkx as nx


NVLINK_BW = 300  # GB/s
A100_IB_BW = 25  # GB/s


def a100_topology(nnode: int):
    G = nx.DiGraph()
    compute_nodes = []
    for a in range(nnode):
        for b in range(8):
            compute_nodes.append((a, "GPU", b))
            # NVSwitch
            G.add_edge((a, "GPU", b), (a, "NVSwitch", 0), capacity=NVLINK_BW)
            G.add_edge((a, "NVSwitch", 0), (a, "GPU", b), capacity=NVLINK_BW)
            if nnode > 1:
                # mlx5 nics
                G.add_edge((a, "GPU", b), (a, "mlx5", b), capacity=A100_IB_BW)
                G.add_edge((a, "mlx5", b), (a, "GPU", b), capacity=A100_IB_BW)
        if nnode > 1:
            # IB Switch: separate IBs for odd/even channel mlx5
            for b in range(8):
                G.add_edge((-1, "IB", b % 2), (a, "mlx5", b), capacity=A100_IB_BW)
                G.add_edge((a, "mlx5", b), (-1, "IB", b % 2), capacity=A100_IB_BW)
    return G, compute_nodes


xGMI_BW = 50
MI250_IB_BW = 16
MI250_ADJ_MAT = [
    [0, 4, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 4, 0],
]


def mi250_topology(nnode: int):
    G = nx.DiGraph()
    compute_nodes = []
    for a in range(nnode):
        for b in range(16):
            for c in range(16):
                if MI250_ADJ_MAT[b][c] > 0:
                    G.add_edge((a, "GPU", b), (a, "GPU", c), capacity=MI250_ADJ_MAT[b][c] * xGMI_BW)
                    G.add_edge((a, "GPU", c), (a, "GPU", b), capacity=MI250_ADJ_MAT[b][c] * xGMI_BW)
            compute_nodes.append((a, "GPU", b))
            if nnode > 1:
                G.add_edge((-1, "IB", 0), (a, "GPU", b), capacity=MI250_IB_BW)
                G.add_edge((a, "GPU", b), (-1, "IB", 0), capacity=MI250_IB_BW)
    return G, compute_nodes
