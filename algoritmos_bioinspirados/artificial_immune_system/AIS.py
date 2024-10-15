import numpy as np 
class Antibody:
    def __init__(self, dim, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dim)
        self.affinity = float('inf')  # Afinidad inicial muy baja (alta distancia a los datos)

class AIS:
    def __init__(self, n_antibodies, dim, lower_bound, upper_bound, max_iter=100):
        self.n_antibodies = n_antibodies
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.antibodies = [Antibody(dim, lower_bound, upper_bound) for _ in range(n_antibodies)]
        self.best_antibody = None
        
    def calculate_affinity(self, antibody, data, labels):
        # Calculamos la afinidad (distancia a los puntos de fraude)
        fraud_data = data[labels == 1]
        distances = np.linalg.norm(fraud_data - antibody.position, axis=1)
        return np.mean(distances)  # Promedio de las distancias a las transacciones fraudulentas

    def mutate(self, antibody):
        # Aplicar mutación a los anticuerpos
        mutation_strength = np.random.uniform(-1, 1, self.dim)
        new_position = antibody.position + mutation_strength
        new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
        return new_position

    def optimize(self, data, labels):
        for iteration in range(self.max_iter):
            # Evaluar la afinidad de cada anticuerpo
            for antibody in self.antibodies:
                antibody.affinity = self.calculate_affinity(antibody, data, labels)
            
            # Selección del mejor anticuerpo
            self.best_antibody = min(self.antibodies, key=lambda ab: ab.affinity)
            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Affinity: {self.best_antibody.affinity}")
            
            # Reproducir y mutar los anticuerpos
            for antibody in self.antibodies:
                antibody.position = self.mutate(self.best_antibody)  # Mutar basado en el mejor anticuerpo
            
        return self.best_antibody
