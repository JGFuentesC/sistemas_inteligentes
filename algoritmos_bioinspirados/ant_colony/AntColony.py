import networkx as nx
import numpy as np

class AntColony:
    def __init__(self, graph, n_ants, n_iterations, decay, alpha=1, beta=1):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-10

        # Precalcular información de las aristas
        self.edge_data = {}
        for edge in self.graph.edges:
            sorted_edge = tuple(sorted(edge))
            self.edge_data[sorted_edge] = {
                'pheromone': 1.0,
                'weight': self.graph[edge[0]][edge[1]]['weight']
            }

    def run(self, start_node, end_node):
        all_time_shortest_path = None
        shortest_path_length = np.inf

        for _ in range(self.n_iterations):
            all_paths = self.generate_all_paths(start_node)
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])

            if shortest_path[1] < shortest_path_length:
                all_time_shortest_path = shortest_path
                shortest_path_length = shortest_path[1]

            # Decaer feromonas
            for edge_data in self.edge_data.values():
                edge_data['pheromone'] *= self.decay

        return all_time_shortest_path

    def spread_pheromone(self, all_paths):
        for path, dist in all_paths:
            for edge in path:
                sorted_edge = tuple(sorted(edge))
                self.edge_data[sorted_edge]['pheromone'] += 1.0 / dist

    def generate_all_paths(self, start_node):
        return [(path, self.calculate_path_distance(path)) 
                for path in [self.generate_path(start_node) for _ in range(self.n_ants)]]

    def generate_path(self, start_node):
        path = []
        visited = set([start_node])
        current_node = start_node

        while len(visited) < len(self.graph.nodes):
            next_node = self.pick_move(current_node, visited)
            if next_node is None:
                break
            path.append((current_node, next_node))
            visited.add(next_node)
            current_node = next_node
        return path

    def pick_move(self, current_node, visited):
        neighbors = [neighbor for neighbor in self.graph.neighbors(current_node) if neighbor not in visited]
        if not neighbors:
            return None

        pheromones = np.array([self.edge_data[tuple(sorted((current_node, neighbor)))]['pheromone'] for neighbor in neighbors])
        distances = np.array([self.edge_data[tuple(sorted((current_node, neighbor)))]['weight'] for neighbor in neighbors])
        
        distances[distances == 0] = self.epsilon
        probabilities = (pheromones ** self.alpha) * ((1.0 / distances) ** self.beta)
        
        if probabilities.sum() == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)
        
        probabilities /= probabilities.sum()

        chosen_neighbor = np.random.choice(neighbors, p=probabilities)
        return chosen_neighbor

    def calculate_path_distance(self, path):
        return sum(self.edge_data[tuple(sorted(edge))]['weight'] for edge in path)