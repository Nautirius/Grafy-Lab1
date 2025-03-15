import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class SimpleGraph:
    def __init__(self, adjacency_matrix=None):
        self.graph = nx.Graph()
        if adjacency_matrix is not None:
            self.num_vertices = len(adjacency_matrix)
            for i in range(self.num_vertices):
                self.graph.add_node(i)
            for i in range(self.num_vertices):
                for j in range(i + 1, self.num_vertices):
                    if adjacency_matrix[i][j] == 1:
                        self.graph.add_edge(i, j)
        else:
            self.num_vertices = 0

    @staticmethod
    def from_adjacency_matrix(matrix):
        return SimpleGraph(matrix)

    @staticmethod
    def from_adjacency_list(adj_list):
        graph = SimpleGraph()
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                graph.graph.add_edge(node, neighbor)
        graph.num_vertices = len(graph.graph.nodes)
        return graph

    @staticmethod
    def from_incidence_matrix(inc_matrix):
        graph = SimpleGraph()
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
        return {node: list(neighbors) for node, neighbors in self.graph.adjacency()}

    # def visualize(self):
    #     pos = nx.circular_layout(self.graph)
    #     nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    #     plt.show()

    def visualize(self):
        plt.figure(figsize=(6, 6))
        x0, y0, r = 0, 0, 5
        alpha = 2 * np.pi / self.num_vertices
        positions = {i: (x0 + r * np.cos(i * alpha), y0 + r * np.sin(i * alpha)) for i in self.graph.nodes}

        circle = plt.Circle((x0, y0), r, color='black', fill=False, linestyle='dashed')
        plt.gca().add_patch(circle)

        for u, v in self.graph.edges:
            x_values = [positions[u][0], positions[v][0]]
            y_values = [positions[u][1], positions[v][1]]
            plt.plot(x_values, y_values, 'gray')

        for i, (x, y) in positions.items():
            plt.scatter(x, y, color='lightblue', edgecolors='black', s=300)
            plt.text(x, y, str(i), fontsize=12, ha='center', va='center')

        plt.axis('off')
        plt.xlim(x0 - r - 1, x0 + r + 1)
        plt.ylim(y0 - r - 1, y0 + r + 1)
        plt.show()

    @staticmethod
    def random_graph_edges(n, l):
        graph = nx.Graph()
        nodes = list(range(n))
        graph.add_nodes_from(nodes)
        edges = set()
        while len(edges) < l:
            u, v = random.sample(nodes, 2)
            if (u, v) not in edges and (v, u) not in edges:
                edges.add((u, v))
        graph.add_edges_from(edges)
        simple_graph = SimpleGraph()
        simple_graph.graph = graph
        simple_graph.num_vertices = n
        return simple_graph

    @staticmethod
    def random_graph_probability(n, p):
        graph = nx.Graph()
        nodes = list(range(n))
        graph.add_nodes_from(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    graph.add_edge(i, j)
        simple_graph = SimpleGraph()
        simple_graph.graph = graph
        simple_graph.num_vertices = n
        return simple_graph


def test_graph():
    print("Testing SimpleGraph class...")

    adjacency_matrix = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]

    graph = SimpleGraph.from_adjacency_matrix(adjacency_matrix)
    print("Adjacency List:", graph.to_adjacency_list())
    print("Incidence Matrix:", graph.to_incidence_matrix())
    print("Adjacency Matrix:", graph.to_adjacency_matrix())
    graph.visualize()

    adjacency_list = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2]
    }
    graph_from_list = SimpleGraph.from_adjacency_list(adjacency_list)
    print("Adjacency Matrix from List:", graph_from_list.to_adjacency_matrix())
    graph_from_list.visualize()

    incidence_matrix = [
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1]
    ]
    graph_from_inc = SimpleGraph.from_incidence_matrix(incidence_matrix)
    print("Adjacency Matrix from Incidence Matrix:", graph_from_inc.to_adjacency_matrix())
    graph_from_inc.visualize()

    print("Generating random graph G(n, l)...")
    rand_graph_edges = SimpleGraph.random_graph_edges(20, 190)
    rand_graph_edges.visualize()

    print("Generating random graph G(n, p)...")
    rand_graph_prob = SimpleGraph.random_graph_probability(20, 0.5)
    rand_graph_prob.visualize()


if __name__ == "__main__":
    test_graph()
