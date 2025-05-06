import numpy as np
import networkx as nx


class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.num_vertices = 0

    @staticmethod
    def from_adjacency_matrix(adj_matrix):
        graph = Graph()
        graph.num_vertices = len(adj_matrix)
        for i in range(graph.num_vertices):
            graph.graph.add_node(i)
        for i in range(graph.num_vertices):
            for j in range(i + 1, graph.num_vertices):
                if adj_matrix[i][j] == 1:
                    graph.graph.add_edge(i, j)
        return graph

    @staticmethod
    def from_adjacency_list(adj_list):
        graph = Graph()
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                graph.graph.add_edge(node, neighbor)
        graph.num_vertices = len(graph.graph.nodes)
        return graph

    @staticmethod
    def from_incidence_matrix(inc_matrix):
        graph = Graph()
        num_vertices = len(inc_matrix)
        num_edges = len(inc_matrix[0]) if num_vertices > 0 else 0

        for col in range(num_edges):
            vertices = [row for row in range(num_vertices) if inc_matrix[row][col] == 1]
            if len(vertices) == 2:
                u, v = vertices
                graph.graph.add_edge(u, v)
        graph.num_vertices = len(graph.graph.nodes)
        return graph

    def to_adjacency_matrix(self):
        matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=int)
        for i, j in self.graph.edges:
            matrix[i][j] = 1
            matrix[j][i] = 1
        return matrix.tolist()

    def to_incidence_matrix(self):
        edges = list(self.graph.edges)
        incidence_matrix = np.zeros((self.num_vertices, len(edges)), dtype=int)
        for col, (i, j) in enumerate(edges):
            incidence_matrix[i][col] = 1
            incidence_matrix[j][col] = 1
        return incidence_matrix.tolist()

    def to_adjacency_list(self):
        return {node: list(self.graph.neighbors(node)) for node in self.graph.nodes}


if __name__ == "__main__":

    # z macierzy sąsiedztwa
    adj_matrix = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
    g = Graph.from_adjacency_matrix(adj_matrix)
    print("Lista sąsiedztwa:", g.to_adjacency_list())

    print("Macierz incydencji:")
    for row in g.to_incidence_matrix():
        print(row)

    print("")

    # z listy sąsiedztwa
    adj_list = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2]
    }
    g2 = Graph.from_adjacency_list(adj_list)

    print("Macierz sąsiedztwa:")
    for row in g2.to_adjacency_matrix():
        print(row)

    print("Macierz incydencji:")
    for row in g2.to_incidence_matrix():
        print(row)

    print("")

    # z macierzy incydencji
    inc_matrix = [
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1]
    ]
    g3 = Graph.from_incidence_matrix(inc_matrix)

    print("Macierz sąsiedztwa:")
    for row in g3.to_adjacency_matrix():
        print(row)

    print("Lista sąsiedztwa:")
    print(g3.to_adjacency_list())
