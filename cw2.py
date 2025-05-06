import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class SimpleGraphVisualizer:
    @staticmethod
    def visualize(graph):
        num_vertices = len(graph.nodes)
        if num_vertices == 0:
            print("Graf jest pusty - brak wierzchołków do wizualizacji")
            return

        plt.figure(figsize=(6, 6))
        x0, y0, r = 0, 0, 5
        alpha = 2 * np.pi / num_vertices
        positions = {
            i: (x0 + r * np.cos(i * alpha), y0 + r * np.sin(i * alpha)) for i in graph.nodes
        }

        circle = plt.Circle((x0, y0), r, color='black', fill=False, linestyle='dashed')
        plt.gca().add_patch(circle)

        for u, v in graph.edges:
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


if __name__ == "__main__":
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    SimpleGraphVisualizer.visualize(g)
