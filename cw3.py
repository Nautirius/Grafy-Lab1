import random
import networkx as nx
from cw2 import SimpleGraphVisualizer


class RandomGraphGenerator:
    @staticmethod
    def generate_edges(n, l):
        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        # wszystkie możliwe unikalne krawędzie
        all_possible_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                all_possible_edges.append((u, v))

        # losowy wybór l krawędzi bez powtórzeń
        chosen_edges = random.sample(all_possible_edges, l)

        graph.add_edges_from(chosen_edges)
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
