# Streaming-Sched: Streaming Scheduling for Dataflow Architectures.
# Copyright (c) 2023 ETH Zurich. All rights reserved.
# See LICENSE for license information.
'''
    Utilities for graph manipulation/info
'''

import networkx as nx
import json

# from networkx.readwrite import json_graph


def get_source_nodes(dag: nx.DiGraph) -> list:
    '''
    Returns the source nodes of a given graph
    :param dag:
    :return:
    '''
    return [node for node in dag.nodes if dag.in_degree(node) == 0]


def get_sink_nodes(dag: nx.DiGraph) -> list:
    '''
    Returns the sink nodes of a given graph
    :param dag:
    :return:
    '''
    return [node for node in dag.nodes if dag.out_degree(node) == 0]


def is_edge_streaming_explicit(dag, src, dst):
    '''
    Returns whether the edge from src to dstis streaming
    :return:
    '''
    return "stream" in dag.edges[(src, dst)] and dag.edges[(src, dst)]['stream']


def is_edge_streaming(edge):
    '''
    Returns whether a given edge is streaming
    :param: edge with data
    :return:
    '''
    return "stream" in edge[2] and edge[2]["stream"]


def get_pseudo_sink(dag):
    '''
    Returns the pseudo sink if any
    :param dag:
    :return:
    '''

    sink = get_sink_nodes(dag)[0]
    if 'pseudo' in dag.nodes[sink] and dag.nodes[sink]['pseudo']:
        return sink
    else:
        return None


def set_all_edges_type(dag: nx.DiGraph, streaming, exclude_pseudo_nodes=True):
    '''
    Set all edges type to streaming (streaming == True) or not streaming (streaming == False).
    When requested, excludes pseudo source/sink nodes
    :param dag:
    :param exclude_pseudo_nodes:
    :return:
    '''

    if exclude_pseudo_nodes:
        source_node = get_source_nodes(dag)[0]
        sink_node = get_pseudo_sink(dag)
    else:
        source_node, sink_node = None, None

    # get edges that do not touch pseudo root
    for e in dag.edges(data=True):
        if e[0] != source_node and e[1] != sink_node:
            e[2]['stream'] = streaming


def is_all_streaming_path(dag: nx.DiGraph, path: list):
    '''
        Checks whether the given path is a streaming path.
        The path is given as a list of nodes
    '''

    src = path[0]
    for dst in path[1:]:
        if 'stream' not in dag.edges[(src, dst)] or dag.edges[(src, dst)]['stream'] == False:
            return False
        src = dst

    return True


def save_to_file(dag: nx.DiGraph, path: str):
    '''
    Saves the DAG to JSON file with attributes. Skips streaming intervals.
    Note: order of the edges may be different.
    '''
    data = nx.readwrite.json_graph.node_link_data(dag)

    # remove all streaming intervals (that are fractions)
    for edge in data['links']:
        if 'streaming_interval' in edge:
            del edge['streaming_interval']

    # remove node ratios
    for node in data['nodes']:
        if 'ratio' in node:
            del node['ratio']

    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def read_from_file(path: str):
    ''' 
    Reads a DAG from JSON file
    '''
    with open(path) as json_file:
        data = json.load(json_file)

        return nx.readwrite.json_graph.node_link_graph(data)


def build_nx(dag):
    '''
        Generates the sequence of nx ops to build the given DAG
    '''

    for src in nx.algorithms.topological_sort(dag):

        for _, dst, data in dag.out_edges(src, data=True):
            is_streaming = is_edge_streaming_explicit(dag, src, dst)
            print(f"dag.add_edge({src}, {dst}, weight = {data['weight']}, stream = {is_streaming})")

    # for src, dst, data in dag.edges(data=True):
    #     is_streaming = is_edge_streaming_explicit(dag, src, dst)
    #     print(f"dag.add_edge({src}, {dst}, weight = {data['weight']}, stream = {is_streaming})")


def build_restricted_nx(dag, set_of_nodes, buffer_nodes=set()):
    '''
        Generates the sequence of nx ops to build the given DAG
        considering a restricted set of dags
    '''

    # scale nodes
    mapping = dict()
    i = 1
    new_buffer_nodes = set()
    for node in set_of_nodes:
        mapping[node] = i
        if node in buffer_nodes:
            new_buffer_nodes.add(i)
        if 'label' in dag.nodes[node]:
            print(f"dag.add_node({i}, label='{dag.nodes[node]['label']}')")
        i += 1
    pseudo_root = i

    # as an alternative, in case there is something related to the
    # edge ordering...
    # for src, dst, data in dag.edges(data=True):

    for src in nx.algorithms.topological_sort(dag):

        for src, dst, data in dag.out_edges(src, data=True):

            if src in set_of_nodes or dst in set_of_nodes:
                is_streaming = is_edge_streaming_explicit(dag, src, dst)

                if src not in set_of_nodes:
                    new_src = 0  # pseudo-root
                else:
                    new_src = mapping[src]

                if dst not in set_of_nodes:
                    new_dst = pseudo_root
                else:
                    new_dst = mapping[dst]

                print(f"dag.add_edge({new_src}, {new_dst}, weight = {data['weight']}, stream = {is_streaming})")

    print(f"dag.add_node(0)")
    print(f"dag.add_node({pseudo_root}, pseudo=True)")
    print("Buffer nodes: ", new_buffer_nodes)


def get_common_ancestors(graph: nx.Graph, start, end, nodelist):
    # Traverse back the nodelist until you found the end node
    assert nodelist[-1] == start
    ancestors = [start]
    prev = start
    for node in nodelist[::-1]:
        if graph.has_edge(prev, node):
            ancestors.append(node)
        if node == end:
            break
        prev = node

    print("Ancestors of ", start, ": ", ancestors)


def _mark_ancestors(graph: nx.Graph, u, v):
    """
    
    Marks the ancestors of u and v in the DFS tree, until a common ancestor is found.
    Used to signal that an undirected cycle exists between u and v

    """

    # TODO prone to optimization

    # find all the ancestors of u and v
    u_ancestors = [u]
    parent = graph.nodes[u]['parent']
    while parent != -1:
        u_ancestors.append(parent)
        parent = graph.nodes[parent]['parent']

    v_ancestors = []
    parent = v
    while parent != -1 and parent not in u_ancestors:
        v_ancestors.append(parent)
        parent = graph.nodes[parent]['parent']

    # parent now points to the first common element: remove the others from u ancestors
    index = u_ancestors.index(parent)
    u_ancestors = u_ancestors[0:index]

    # start marking:
    graph.nodes[parent]['marked'] = True
    # print("\tMarked: ", parent)
    for node in u_ancestors:
        # The original idea was to stop marking when we found an already marked node.
        # However this is not ok as there can be still non-marked node till we reach the
        # common ancestor (see tests_graph.py, test_medium_undirected_cycles)
        # if 'marked' in graph.nodes[node] and graph.nodes[node]['marked']:
        #     break
        graph.nodes[node]['marked'] = True
        # print("\tMarked: ", node)

    for node in v_ancestors:
        # if 'marked' in graph.nodes[node] and graph.nodes[node]['marked']:
        #     break
        graph.nodes[node]['marked'] = True
        # print("\tMarked: ", node)


def _get_marked_components(graph: nx.Graph, start):
    """
    Returns the marked components in the graph
    """

    # keep track of the marked components
    # TODO optimize the way in which we detect different components
    components = [set()]

    # go in the same DFS order: as soon as we find a gap, then another cycle starts
    visited = []
    stack = [start]
    vertex = start
    # components_by_node = dict()
    while stack:
        # print("Stack: ", stack)
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            node = vertex
            if 'marked' in graph.nodes[node] and graph.nodes[node]['marked']:
                # print("Node: ", node, " parent: ", graph.nodes[node]['parent'], " comp: ", components[-1])
                if len(components[-1]) > 0 and graph.nodes[node]['parent'] not in components[-1]:
                    # this is a new component
                    # print("Create a new component")
                    components.append(set())
                components[-1].add(node)

            for n in graph.neighbors(vertex):
                if n not in visited:
                    graph.nodes[n]['parent'] = vertex
                    stack.append(n)

    #         # stupid version
    #         # for each node keep track of the component at which it belongs by looking at the parent
    #         parent = graph.nodes[vertex]['parent']
    #         if parent != -1:
    #             if parent in components_by_node:
    #                 parent_comp = components_by_node[parent]
    #             else:
    #                 parent_comp = set()
    #             parent_comp.add(vertex)
    #             components_by_node[vertex] = parent_comp

    return components


def get_undirected_cycles(dag: nx.DiGraph, pseudo_source: int = -1):
    """
    Generates a list of undirected cycles contained in the given direct DAG.

    If a cycle is contained within another, it will return a single node set containing
    the nodes of both cycles

    :param dag: direct graph
    :return: list of cycles (list of set of nodes)
    """

    # Assumes weakly connected graph? Don't think so

    # The base of the algorithm is a DFS of the undirected graph
    # Every time a (u,v) back edge is found, we mark all the nodes in the DFS tree
    # that are ancestors of u and v, until we reach a common ancestor. These nodes are
    # part of a cycle
    # In the end, we compute all the cycles as the marked connected components of the DFS tree.

    graph = dag.to_undirected()

    # remove the pseudo source
    if pseudo_source != -1:
        if graph.has_node(pseudo_source):
            graph.remove_node(pseudo_source)

    cycles = []
    # The graph may have multiple connected components: loop over all of them
    for subg in nx.algorithms.connected_components(graph):
        subgraph = graph.subgraph(subg)

        # start from a random node
        start = list(subg)[0]

        visited = []
        stack = [start]
        vertex = start

        # We store the info about the DFS tree as nodes attributes
        subgraph.nodes[vertex]['parent'] = -1
        while stack:
            # print("Stack: ", stack)
            vertex = stack.pop()
            parent = subgraph.nodes[vertex]['parent']
            # print("Popped:", vertex, " parent: ", parent)
            if vertex not in visited:
                visited.append(vertex)

                for n in subgraph.neighbors(vertex):
                    # safety check: no self-loops

                    assert n != vertex

                    if n in visited:

                        if n != parent:
                            # This is a back edge, mark ancestors
                            # print("Back edge ", vertex, " -> ", n, " parent: ", parent)
                            _mark_ancestors(subgraph, vertex, n)
                    else:
                        subgraph.nodes[n]['parent'] = vertex
                        stack.append(n)
                # print("\tNew stack: ", stack)
                # stack.extend(subgraph[vertex] - visited)
            # else:
            #     print("Already visited: ", vertex)

        cycles.extend(_get_marked_components(subgraph, start))

    return cycles


def get_type_of_nodes(dag: nx.DiGraph, source: int, pseudo_sink: int, buffer_nodes=set()):
    '''
    Counts the number of elwise, upsampler and downsampler nodes in the DAG
    Source and pseudo sink are not considered
    '''
    elwise = 0
    upsampler = 0
    downsampler = 0
    for n in dag.nodes():
        if n == source or n == pseudo_sink or n in buffer_nodes:
            continue

        read_data = list(dag.in_edges(n, data=True))[0][2]['weight']

        if dag.out_degree(n) == 0:
            produced_data = read_data
        else:
            produced_data = list(dag.out_edges(n, data=True))[0][2]['weight']

        if read_data == produced_data:
            elwise += 1
        elif read_data < produced_data:
            upsampler += 1
        else:
            downsampler += 1

    total = elwise + upsampler + downsampler
    print(
        f"Elwise: {elwise} ({(elwise/total) *100:.1f} %), Downsampler: {downsampler} ({(downsampler/total) *100:.1f} %), Upsampler: {upsampler} ({(upsampler/total) *100:.1f} %)"
    )


def get_average_edge_weight(dag: nx.DiGraph):
    '''
    Computes the average edge weight of a given graph
    (considering also pseudo nodes)
    '''
    max_volume = 0
    volume = 0
    for _, _, data in dag.edges(data=True):
        volume += data['weight']
        max_volume = max(max_volume, data['weight'])

    # print("Max volume: ", max_volume)
    return volume / dag.number_of_edges()
