import random
import networkx as nx
from cw2 import SimpleGraphVisualizer


class RandomGraphGenerator:
    @staticmethod
    def generate_edges(n, l):
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        edges = set()
        while len(edges) < l:
            u, v = random.sample(range(n), 2)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.add((u, v))
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def generate_probability(n, p):
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    graph.add_edge(i, j)
        return graph


if __name__ == "__main__":
    print("Random G(n, l)")
    g1 = RandomGraphGenerator.generate_edges(10, 15)
    SimpleGraphVisualizer.visualize(g1)

    print("Random G(n, p)")
    g2 = RandomGraphGenerator.generate_probability(10, 0.3)
    SimpleGraphVisualizer.visualize(g2)
