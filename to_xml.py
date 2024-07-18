import xml.etree.ElementTree as ET
import networkx as nx


def _get_unused_chan(channel_usage: dict, links: list, odd_even: int = None) -> int:
    assert odd_even is None or odd_even == 0 or odd_even == 1
    used_chans = set()
    chan = None
    for a, b in links:
        if (a, b) not in channel_usage:
            channel_usage[a, b] = []
        else:
            used_chans.update(channel_usage[a, b])
    if len(used_chans) != 0:
        for c in range(max(used_chans) + 3):
            if c not in used_chans and (odd_even is None or c % 2 == odd_even):
                chan = c
                break
    else:
        chan = 0 if odd_even is None else odd_even
    assert chan is not None and chan not in used_chans
    for a, b in links:
        assert chan not in channel_usage[a, b]
        channel_usage[a, b].append(chan)
    return chan


def construct_algo_allreduce(Ts, Cs, k, nodes, ninstance, proto="Simple", inplace=True, nics_str=None):
    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        assert C == 1
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    chunks_per_shard = k
    nchunksperloop = ninstance * chunks_per_shard * len(nodes)

    algo = ET.Element("algo", attrib={
        "name": f"pipeline schedule (k{k},inst{ninstance})",
        "proto": proto,
        "nchunksperloop": str(nchunksperloop),
        "ngpus": str(len(nodes)),
        "coll": "allreduce",
        "inplace": "1" if inplace else "0",
        "outofplace": "0" if inplace else "1",
        "minBytes": "0",
        "maxBytes": "1099511627776"
    })
    if nics_str is not None:
        for l in nics_str.split('\n'):
            algo.append(ET.Comment(l))

    gpus = {}
    for gpu_id in sorted(nodes):
        gpus[gpu_id] = ET.SubElement(algo, "gpu", attrib={
            "id": str(gpu_id),
            "i_chunks": str(0 if inplace else nchunksperloop),
            "o_chunks": str(nchunksperloop),
            "s_chunks": str(0),
        })

    channel_usage = {}
    threads = {}
    nthreads = {u: 0 for u in nodes}
    for inst in range(ninstance):
        for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
            es = []
            odd_even_chan = {}
            for p in ps:
                assert len(p) == 1 and len(p[0]) == 3
                a, b, ch = p[0]
                es.append((a, b))
                odd_even_chan[a, b] = ch
            test_G = nx.DiGraph()
            test_G.add_edges_from(es)

            chunk = u * chunks_per_shard * ninstance + inst * chunks_per_shard + chunk_starts[u, i]

            node_properties = {a: {
                'parent': None, 'main_child': -1, 'children': [], 'parent_chan': None, 'children_chans': {},
            } for a in nodes}
            for a, b in es:
                assert node_properties[b]['parent'] is None
                node_properties[b]['parent'] = a

            paths = nx.single_source_shortest_path(test_G, u)
            visited = set()
            for dest in sorted(nodes, key=lambda d: len(paths[d]), reverse=True):
                if test_G.out_degree(dest) > 0:
                    continue
                path = paths[dest]
                for idx in range(len(path) - 1):
                    if path[idx + 1] not in node_properties[path[idx]]['children']:
                        node_properties[path[idx]]['children'].append(path[idx + 1])
                start = 0
                while path[start] in visited:
                    start += 1
                if start > 0:
                    start -= 1
                visited.update(path)
                if start == 0 and node_properties[path[0]]['main_child'] == -1:
                    node_properties[path[0]]['main_child'] = path[1]

                ch = None
                channel_segments = []
                seg = [path[start]]
                for a, b in zip(path[start:-1], path[start + 1:]):
                    if odd_even_chan[a, b] is not None:
                        if ch is not None and ch != odd_even_chan[a, b]:
                            channel_segments.append((seg, ch))
                            seg = [a]
                        ch = odd_even_chan[a, b]
                    seg.append(b)
                channel_segments.append((seg, ch))

                for seg, ch in channel_segments:
                    assert len(seg) > 1
                    channel = _get_unused_chan(
                        channel_usage,
                        list(zip(seg[:-1], seg[1:])) + list(zip(seg[1:], seg[:-1])),
                        ch
                    )
                    for idx in range(len(seg) - 1):
                        node_properties[seg[idx]]['children_chans'][seg[idx + 1]] = channel
                        node_properties[seg[idx + 1]]['parent_chan'] = channel
                        if idx > 0 and node_properties[seg[idx]]['main_child'] == -1:
                            node_properties[seg[idx]]['main_child'] = seg[idx + 1]
                    node_properties[seg[-1]]['parent_chan'] = channel

            for node in nodes:
                rrc_thread_id = None
                for child in reversed(node_properties[node]['children']):
                    if child != node_properties[node]['main_child']:
                        thread_id = nthreads[node]
                        nthreads[node] += 1
                        threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                            "id": str(thread_id),
                            "send": str(-1),
                            "recv": str(child),
                            "chan": str(node_properties[node]['children_chans'][child]),
                        })
                        threads[node, thread_id].append(
                            ET.Comment(f"RS, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                        ET.SubElement(threads[node, thread_id], "step", attrib={
                            "s": str(0),
                            "type": "rrc",
                            "srcbuf": "o" if inplace else "i",
                            "srcoff": str(chunk),
                            "dstbuf": "o" if inplace else "i",
                            "dstoff": str(chunk),
                            "cnt": str(Cs[u, i]),
                            "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                            "deps": str(-1 if rrc_thread_id is None else 0),
                            "hasdep": str(1),
                        })
                        rrc_thread_id = thread_id
                if node_properties[node]['parent'] is None:
                    # root
                    recv_thread_id = nthreads[node]
                    nthreads[node] += 1
                    assert node_properties[node]['main_child'] != -1
                    threads[node, recv_thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(recv_thread_id),
                        "send": str(node_properties[node]['main_child']),
                        "recv": str(node_properties[node]['main_child']),
                        "chan": str(node_properties[node]['children_chans'][node_properties[node]['main_child']]),
                    })
                    threads[node, recv_thread_id].append(
                        ET.Comment(f"ROOT, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, recv_thread_id], "step", attrib={
                        "s": str(0),
                        "type": "rrcs",
                        "srcbuf": "o" if inplace else "i",
                        "srcoff": str(chunk),
                        "dstbuf": "o",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                        "deps": str(-1 if rrc_thread_id is None else 0),
                        "hasdep": str(1 if len(node_properties[node]['children']) > 1 else 0),
                    })
                else:
                    thread_id = nthreads[node]
                    nthreads[node] += 1
                    if len(node_properties[node]['children']) > 0 and node_properties[node]['main_child'] != -1:
                        assert node_properties[node]['parent_chan'] == \
                               node_properties[node]['children_chans'][node_properties[node]['main_child']]
                    threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(thread_id),
                        "send": str(node_properties[node]['parent']),
                        "recv": str(node_properties[node]['main_child']),
                        "chan": str(node_properties[node]['parent_chan']),
                    })
                    threads[node, thread_id].append(
                        ET.Comment(f"RS, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, thread_id], "step", attrib={
                        "s": str(0),
                        # leaf if len(node_properties[node]['children']) = 0
                        "type": "rrs" if len(node_properties[node]['children']) > 0 and
                                         node_properties[node]['main_child'] != -1 else "s",
                        "srcbuf": "o" if inplace else "i",
                        "srcoff": str(chunk),
                        "dstbuf": "o" if inplace else "i",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                        "deps": str(-1 if rrc_thread_id is None else 0),
                        "hasdep": str(0),
                    })
                    recv_thread_id = nthreads[node]
                    nthreads[node] += 1
                    threads[node, recv_thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(recv_thread_id),
                        "send": str(node_properties[node]['main_child']),
                        "recv": str(node_properties[node]['parent']),
                        "chan": str(node_properties[node]['parent_chan']),
                    })
                    threads[node, recv_thread_id].append(
                        ET.Comment(f"AG, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, recv_thread_id], "step", attrib={
                        "s": str(0),
                        # leaf if len(node_properties[node]['children']) = 0
                        "type": "rcs" if len(node_properties[node]['children']) > 0 and
                                         node_properties[node]['main_child'] != -1 else "r",
                        "srcbuf": "o",
                        "srcoff": str(chunk),
                        "dstbuf": "o",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": "-1",
                        "deps": "-1",
                        "hasdep": str(1 if len(node_properties[node]['children']) > 1 else 0),
                    })
                for child in node_properties[node]['children']:
                    if child != node_properties[node]['main_child']:
                        thread_id = nthreads[node]
                        nthreads[node] += 1
                        threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                            "id": str(thread_id),
                            "send": str(child),
                            "recv": str(-1),
                            "chan": str(node_properties[node]['children_chans'][child]),
                        })
                        threads[node, thread_id].append(
                            ET.Comment(f"AG, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                        ET.SubElement(threads[node, thread_id], "step", attrib={
                            "s": str(0),
                            "type": "s",
                            "srcbuf": "o",
                            "srcoff": str(chunk),
                            "dstbuf": "o",
                            "dstoff": str(chunk),
                            "cnt": str(Cs[u, i]),
                            "depid": str(recv_thread_id),
                            "deps": "0",
                            "hasdep": str(0),
                        })
    algo.set("nchannels", str(max(max(cs) for cs in channel_usage.values()) + 1))
    assert max(nthreads.values()) <= 108
    nchannels = max(max(cs) for cs in channel_usage.values()) + 1
    assert nchannels <= 32
    algo.set("nchannels", str(nchannels))
    return algo


def construct_algo_allgather(Ts, Cs, k, nodes, ninstance, proto="Simple", inplace=True, nics_str=None):
    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        assert C == 1
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    chunks_per_shard = k
    nchunksperloop = ninstance * chunks_per_shard * len(nodes)

    algo = ET.Element("algo", attrib={
        "name": f"pipeline schedule (k{k},inst{ninstance})",
        "proto": proto,
        "nchunksperloop": str(nchunksperloop),
        "ngpus": str(len(nodes)),
        "coll": "allgather",
        "inplace": "1" if inplace else "0",
        "outofplace": "0" if inplace else "1",
        "minBytes": "0",
        "maxBytes": "1099511627776"
    })
    if nics_str is not None:
        for l in nics_str.split('\n'):
            algo.append(ET.Comment(l))

    gpus = {}
    for gpu_id in sorted(nodes):
        gpus[gpu_id] = ET.SubElement(algo, "gpu", attrib={
            "id": str(gpu_id),
            "i_chunks": str(0 if inplace else nchunksperloop),
            "o_chunks": str(nchunksperloop),
            "s_chunks": str(0),
        })

    channel_usage = {}
    threads = {}
    nthreads = {u: 0 for u in nodes}
    for inst in range(ninstance):
        for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
            es = []
            odd_even_chan = {}
            for p in ps:
                assert len(p) == 1 and len(p[0]) == 3
                a, b, ch = p[0]
                es.append((a, b))
                odd_even_chan[a, b] = ch
            test_G = nx.DiGraph()
            test_G.add_edges_from(es)

            chunk = u * chunks_per_shard * ninstance + inst * chunks_per_shard + chunk_starts[u, i]

            node_properties = {a: {
                'parent': None, 'main_child': -1, 'children': [], 'parent_chan': None, 'children_chans': {},
            } for a in nodes}
            for a, b in es:
                assert node_properties[b]['parent'] is None
                node_properties[b]['parent'] = a

            paths = nx.single_source_shortest_path(test_G, u)
            visited = set()
            for dest in sorted(nodes, key=lambda d: len(paths[d]), reverse=True):
                if test_G.out_degree(dest) > 0:
                    continue
                path = paths[dest]
                for idx in range(len(path) - 1):
                    if path[idx + 1] not in node_properties[path[idx]]['children']:
                        node_properties[path[idx]]['children'].append(path[idx + 1])
                start = 0
                while path[start] in visited:
                    start += 1
                if start > 0:
                    start -= 1
                visited.update(path)
                if start == 0 and node_properties[path[0]]['main_child'] == -1:
                    node_properties[path[0]]['main_child'] = path[1]

                ch = None
                channel_segments = []
                seg = [path[start]]
                for a, b in zip(path[start:-1], path[start + 1:]):
                    if odd_even_chan[a, b] is not None:
                        if ch is not None and ch != odd_even_chan[a, b]:
                            channel_segments.append((seg, ch))
                            seg = [a]
                        ch = odd_even_chan[a, b]
                    seg.append(b)
                channel_segments.append((seg, ch))

                for seg, ch in channel_segments:
                    assert len(seg) > 1
                    channel = _get_unused_chan(
                        channel_usage,
                        list(zip(seg[:-1], seg[1:])),
                        ch
                    )
                    for idx in range(len(seg) - 1):
                        node_properties[seg[idx]]['children_chans'][seg[idx + 1]] = channel
                        node_properties[seg[idx + 1]]['parent_chan'] = channel
                        if idx > 0 and node_properties[seg[idx]]['main_child'] == -1:
                            node_properties[seg[idx]]['main_child'] = seg[idx + 1]
                    node_properties[seg[-1]]['parent_chan'] = channel

            for node in nodes:
                if node_properties[node]['parent'] is None:
                    # root
                    recv_thread_id = nthreads[node]
                    nthreads[node] += 1
                    assert node_properties[node]['main_child'] != -1
                    threads[node, recv_thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(recv_thread_id),
                        "send": str(node_properties[node]['main_child']),
                        # "recv": str(node_properties[node]['main_child']),
                        "recv": str(-1),
                        "chan": str(node_properties[node]['children_chans'][node_properties[node]['main_child']]),
                    })
                    threads[node, recv_thread_id].append(
                        ET.Comment(f"ROOT, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, recv_thread_id], "step", attrib={
                        "s": str(0),
                        "type": "s",
                        "srcbuf": "o" if inplace else "i",
                        "srcoff": str(chunk),
                        "dstbuf": "o",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": str(-1),
                        "deps": str(-1),
                        "hasdep": str(0),
                    })
                    recv_thread_id = None  # root does not recv
                else:
                    recv_thread_id = nthreads[node]
                    nthreads[node] += 1
                    threads[node, recv_thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(recv_thread_id),
                        "send": str(node_properties[node]['main_child']),
                        "recv": str(node_properties[node]['parent']),
                        "chan": str(node_properties[node]['parent_chan']),
                    })
                    threads[node, recv_thread_id].append(
                        ET.Comment(f"AG, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, recv_thread_id], "step", attrib={
                        "s": str(0),
                        # leaf if len(node_properties[node]['children']) = 0
                        "type": "rcs" if len(node_properties[node]['children']) > 0 and
                                         node_properties[node]['main_child'] != -1 else "r",
                        "srcbuf": "o",
                        "srcoff": str(chunk),
                        "dstbuf": "o",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": "-1",
                        "deps": "-1",
                        "hasdep": str(1 if len(node_properties[node]['children']) > 1 else 0),
                    })
                for child in node_properties[node]['children']:
                    if child != node_properties[node]['main_child']:
                        thread_id = nthreads[node]
                        nthreads[node] += 1
                        threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                            "id": str(thread_id),
                            "send": str(child),
                            "recv": str(-1),
                            "chan": str(node_properties[node]['children_chans'][child]),
                        })
                        threads[node, thread_id].append(
                            ET.Comment(f"AG, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                        ET.SubElement(threads[node, thread_id], "step", attrib={
                            "s": str(0),
                            "type": "s",
                            "srcbuf": "o",
                            "srcoff": str(chunk),
                            "dstbuf": "o",
                            "dstoff": str(chunk),
                            "cnt": str(Cs[u, i]),
                            "depid": str(recv_thread_id if recv_thread_id is not None else -1),
                            "deps": "0" if recv_thread_id is not None else "-1",
                            "hasdep": str(0),
                        })
    assert max(nthreads.values()) <= 108
    nchannels = max(max(cs) for cs in channel_usage.values()) + 1
    assert nchannels <= 32
    algo.set("nchannels", str(nchannels))
    return algo


def construct_algo_reduce_scatter(Ts, Cs, k, nodes, ninstance, is_amd, proto="Simple", inplace=True, nics_str=None):
    chunk_starts = {}
    chunk_counts = {}
    for (u, i), C in sorted(Cs.items(), key=lambda x: x[0]):
        assert C == 1
        if u not in chunk_counts:
            chunk_starts[u, i] = 0
            chunk_counts[u] = C
        else:
            chunk_starts[u, i] = chunk_counts[u]
            chunk_counts[u] += C
    chunks_per_shard = k
    nchunksperloop = ninstance * chunks_per_shard * len(nodes)
    assert inplace

    algo = ET.Element("algo", attrib={
        "name": f"pipeline schedule (k{k},inst{ninstance})",
        "proto": proto,
        "nchunksperloop": str(nchunksperloop),
        "ngpus": str(len(nodes)),
        "coll": "reducescatter" if is_amd else "reduce_scatter",
        "inplace": "1" if inplace else "0",
        "outofplace": "0" if inplace else "1",
        "minBytes": "0",
        "maxBytes": "1099511627776"
    })
    if nics_str is not None:
        for l in nics_str.split('\n'):
            algo.append(ET.Comment(l))

    gpus = {}
    for gpu_id in sorted(nodes):
        gpus[gpu_id] = ET.SubElement(algo, "gpu", attrib={
            "id": str(gpu_id),
            "i_chunks": str(nchunksperloop),
            "o_chunks": str(0 if inplace else nchunksperloop),
            "s_chunks": str(0),
        })

    channel_usage = {}
    threads = {}
    nthreads = {u: 0 for u in nodes}
    for inst in range(ninstance):
        for (u, i), ps in sorted(Ts.items(), key=lambda x: x[0]):
            es = []
            odd_even_chan = {}
            for p in ps:
                assert len(p) == 1 and len(p[0]) == 3
                a, b, ch = p[0]
                es.append((a, b))
                odd_even_chan[a, b] = ch
            test_G = nx.DiGraph()
            test_G.add_edges_from(es)

            chunk = u * chunks_per_shard * ninstance + inst * chunks_per_shard + chunk_starts[u, i]

            node_properties = {a: {
                'parent': None, 'main_child': -1, 'children': [], 'parent_chan': None, 'children_chans': {},
            } for a in nodes}
            for a, b in es:
                assert node_properties[b]['parent'] is None
                node_properties[b]['parent'] = a

            paths = nx.single_source_shortest_path(test_G, u)
            visited = set()
            for dest in sorted(nodes, key=lambda d: len(paths[d]), reverse=True):
                if test_G.out_degree(dest) > 0:
                    continue
                path = paths[dest]
                for idx in range(len(path) - 1):
                    if path[idx + 1] not in node_properties[path[idx]]['children']:
                        node_properties[path[idx]]['children'].append(path[idx + 1])
                start = 0
                while path[start] in visited:
                    start += 1
                if start > 0:
                    start -= 1
                visited.update(path)
                if start == 0 and node_properties[path[0]]['main_child'] == -1:
                    node_properties[path[0]]['main_child'] = path[1]

                ch = None
                channel_segments = []
                seg = [path[start]]
                for a, b in zip(path[start:-1], path[start + 1:]):
                    if odd_even_chan[a, b] is not None:
                        if ch is not None and ch != odd_even_chan[a, b]:
                            channel_segments.append((seg, ch))
                            seg = [a]
                        ch = odd_even_chan[a, b]
                    seg.append(b)
                channel_segments.append((seg, ch))

                for seg, ch in channel_segments:
                    assert len(seg) > 1
                    channel = _get_unused_chan(
                        channel_usage,
                        list(zip(seg[1:], seg[:-1])),
                        ch
                    )
                    for idx in range(len(seg) - 1):
                        node_properties[seg[idx]]['children_chans'][seg[idx + 1]] = channel
                        node_properties[seg[idx + 1]]['parent_chan'] = channel
                        if idx > 0 and node_properties[seg[idx]]['main_child'] == -1:
                            node_properties[seg[idx]]['main_child'] = seg[idx + 1]
                    node_properties[seg[-1]]['parent_chan'] = channel

            for node in nodes:
                rrc_thread_id = None
                for child in reversed(node_properties[node]['children']):
                    if child != node_properties[node]['main_child']:
                        thread_id = nthreads[node]
                        nthreads[node] += 1
                        threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                            "id": str(thread_id),
                            "send": str(-1),
                            "recv": str(child),
                            "chan": str(node_properties[node]['children_chans'][child]),
                        })
                        threads[node, thread_id].append(
                            ET.Comment(f"RS, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                        ET.SubElement(threads[node, thread_id], "step", attrib={
                            "s": str(0),
                            "type": "rrc",
                            "srcbuf": "i",
                            "srcoff": str(chunk),
                            "dstbuf": "i",
                            "dstoff": str(chunk),
                            "cnt": str(Cs[u, i]),
                            "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                            "deps": str(-1 if rrc_thread_id is None else 0),
                            "hasdep": str(1),
                        })
                        rrc_thread_id = thread_id
                if node_properties[node]['parent'] is None:
                    # root
                    recv_thread_id = nthreads[node]
                    nthreads[node] += 1
                    assert node_properties[node]['main_child'] != -1
                    threads[node, recv_thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(recv_thread_id),
                        # "send": str(node_properties[node]['main_child']),
                        "send": str(-1),
                        "recv": str(node_properties[node]['main_child']),
                        "chan": str(node_properties[node]['children_chans'][node_properties[node]['main_child']]),
                    })
                    threads[node, recv_thread_id].append(
                        ET.Comment(f"ROOT, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, recv_thread_id], "step", attrib={
                        "s": str(0),
                        "type": "rrc",
                        "srcbuf": "i",
                        "srcoff": str(chunk),
                        "dstbuf": "i",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                        "deps": str(-1 if rrc_thread_id is None else 0),
                        "hasdep": str(1 if len(node_properties[node]['children']) > 1 else 0),
                    })
                else:
                    thread_id = nthreads[node]
                    nthreads[node] += 1
                    if len(node_properties[node]['children']) > 0 and node_properties[node]['main_child'] != -1:
                        assert node_properties[node]['parent_chan'] == \
                               node_properties[node]['children_chans'][node_properties[node]['main_child']]
                    threads[node, thread_id] = ET.SubElement(gpus[node], "tb", attrib={
                        "id": str(thread_id),
                        "send": str(node_properties[node]['parent']),
                        "recv": str(node_properties[node]['main_child']),
                        "chan": str(node_properties[node]['parent_chan']),
                    })
                    threads[node, thread_id].append(
                        ET.Comment(f"RS, node{node}, root{u}, index{i}, chunks{Cs[u, i]}, inst{inst}"))
                    ET.SubElement(threads[node, thread_id], "step", attrib={
                        "s": str(0),
                        # leaf if len(node_properties[node]['children']) = 0
                        "type": "rrs" if len(node_properties[node]['children']) > 0 and
                                         node_properties[node]['main_child'] != -1 else "s",
                        "srcbuf": "i",
                        "srcoff": str(chunk),
                        "dstbuf": "i",
                        "dstoff": str(chunk),
                        "cnt": str(Cs[u, i]),
                        "depid": str(-1 if rrc_thread_id is None else rrc_thread_id),
                        "deps": str(-1 if rrc_thread_id is None else 0),
                        "hasdep": str(0),
                    })
    assert max(nthreads.values()) <= 108
    nchannels = max(max(cs) for cs in channel_usage.values()) + 1
    assert nchannels <= 32
    algo.set("nchannels", str(nchannels))
    return algo
