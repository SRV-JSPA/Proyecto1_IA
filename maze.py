import math
import heapq
from collections import deque
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import csv
import pandas as pd
import os
    
@dataclass
class ResultadoBusqueda:
    ruta: Optional[List[Tuple[int, int]]]  
    costo: Optional[int]  
    nodos_explorados: int  
    profundidad_ramas: Dict[int, float] 
    
@dataclass
class Maze:
    grid: List[List[str]]

    @classmethod
    def archivo(cls, filename: str):
        with open(filename, 'r') as f:
            reader = csv.reader(f) 
            grid = [list(row) for row in reader]  

        chars_permitidos = {'0', '1', '2', '3'}
        if any(cell not in chars_permitidos for row in grid for cell in row):
            raise ValueError(f"El laberinto contiene caracteres no permitidos: {set(cell for row in grid for cell in row)}")

        return cls(grid)

    def encontrar_inicio_meta(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        inicio, meta = None, None
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == '2':
                    inicio = (i, j)
                elif cell == '3':
                    meta = (i, j)
        return inicio, meta

    def obtener_vecinos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        direcciones = [(-1, 0), (0, 1), (1, 0), (0, -1)]  
        filas, cols = len(self.grid), len(self.grid[0])
        vecinos = []

        for d in direcciones:
            new_i, new_j = pos[0] + d[0], pos[1] + d[1]
            if 0 <= new_i < filas and 0 <= new_j < cols:
                if self.grid[new_i][new_j] in {'0', '2', '3'}:  
                    vecinos.append((new_i, new_j))

        return vecinos

@dataclass
class AlgoritmoBusqueda:
    maze: Maze

    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        raise NotImplementedError
    
    @staticmethod
    def heuristicaa_euclidiana(pos, meta):
        return math.sqrt((pos[0] - meta[0]) ** 2 + (pos[1] - meta[1]) ** 2)

    @staticmethod
    def heuristicaa_manhattan(pos, meta):
        return abs(pos[0] - meta[0]) + abs(pos[1] - meta[1])
    
@dataclass
class BFS(AlgoritmoBusqueda):
    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        queue = deque([(inicio, [inicio])])
        visitados = set([inicio])
        nodos_explorados = 0
        branching_data = {}

        while queue:
            current, ruta = queue.popleft()
            nodos_explorados += 1
            depth = len(ruta) - 1
            children = [n for n in self.maze.obtener_vecinos(current) if n not in visitados]
            branching_data.setdefault(depth, []).append(len(children))

            if current == meta:
                return ResultadoBusqueda(ruta, len(ruta) - 1, nodos_explorados, {d: sum(v)/len(v) for d, v in branching_data.items()})

            for child in children:
                visitados.add(child)
                queue.append((child, ruta + [child]))

        return ResultadoBusqueda(None, None, nodos_explorados, {})
    
@dataclass
class DFS(AlgoritmoBusqueda):
    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        stack = [(inicio, [inicio])]
        visitados = set([inicio])
        nodos_explorados = 0
        branching_data = {}

        while stack:
            current, ruta = stack.pop()
            nodos_explorados += 1
            depth = len(ruta) - 1
            children = [n for n in self.maze.obtener_vecinos(current) if n not in visitados]
            branching_data.setdefault(depth, []).append(len(children))

            if current == meta:
                return ResultadoBusqueda(ruta, len(ruta) - 1, nodos_explorados, {d: sum(v)/len(v) for d, v in branching_data.items()})

            for child in children:
                visitados.add(child)
                stack.append((child, ruta + [child]))

        return ResultadoBusqueda(None, None, nodos_explorados, {})
    
@dataclass
class UCS(AlgoritmoBusqueda):
    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        queue = []
        heapq.heappush(queue, (0, inicio, [inicio]))
        visitados = {inicio: 0}
        nodos_explorados = 0
        branching_data = {}

        while queue:
            costo, current, ruta = heapq.heappop(queue)
            nodos_explorados += 1
            depth = len(ruta) - 1
            children = [n for n in self.maze.obtener_vecinos(current) if n not in visitados]
            branching_data.setdefault(depth, []).append(len(children))

            if current == meta:
                return ResultadoBusqueda(ruta, costo, nodos_explorados, {d: sum(v)/len(v) for d, v in branching_data.items()})

            for child in children:
                new_costo = costo + 1
                visitados[child] = new_costo
                heapq.heappush(queue, (new_costo, child, ruta + [child]))

        return ResultadoBusqueda(None, None, nodos_explorados, {})
    
@dataclass
class Greedy(AlgoritmoBusqueda):
    heuristica: callable

    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        queue = []
        heapq.heappush(queue, (self.heuristica(inicio, meta), inicio, [inicio]))
        visitados = set([inicio])
        nodos_explorados = 0
        branching_data = {}

        while queue:
            _, current, ruta = heapq.heappop(queue)
            nodos_explorados += 1
            depth = len(ruta) - 1
            children = [n for n in self.maze.obtener_vecinos(current) if n not in visitados]
            branching_data.setdefault(depth, []).append(len(children))

            if current == meta:
                return ResultadoBusqueda(ruta, len(ruta) - 1, nodos_explorados, {d: sum(v)/len(v) for d, v in branching_data.items()})

            for child in children:
                visitados.add(child)
                heapq.heappush(queue, (self.heuristica(child, meta), child, ruta + [child]))

        return ResultadoBusqueda(None, None, nodos_explorados, {})
    
@dataclass
class AStar(AlgoritmoBusqueda):
    heuristica: callable

    def busqueda(self, inicio: Tuple[int, int], meta: Tuple[int, int]) -> ResultadoBusqueda:
        queue = []
        heapq.heappush(queue, (self.heuristica(inicio, meta), 0, inicio, [inicio]))
        visitados = {inicio: 0}
        nodos_explorados = 0
        branching_data = {}

        while queue:
            f, costo, current, ruta = heapq.heappop(queue)
            nodos_explorados += 1
            depth = len(ruta) - 1
            children = [n for n in self.maze.obtener_vecinos(current) if n not in visitados]
            branching_data.setdefault(depth, []).append(len(children))

            if current == meta:
                return ResultadoBusqueda(ruta, costo, nodos_explorados, {d: sum(v)/len(v) for d, v in branching_data.items()})

            for child in children:
                new_costo = costo + 1
                visitados[child] = new_costo
                heapq.heappush(queue, (new_costo + self.heuristica(child, meta), new_costo, child, ruta + [child]))

        return ResultadoBusqueda(None, None, nodos_explorados, {})

@dataclass
class MazeSolver:
    maze: Maze

    def graficar_solucion(self, result: ResultadoBusqueda, title: str):
        mapeo = {'0': 0, '1': 1, '2': 2, '3': 3}
        maze_array = np.array([[mapeo.get(cell, 0) for cell in row] for row in self.maze.grid])
        cmap = ListedColormap(['white', 'black', 'green', 'blue'])

        plt.figure(figsize=(10, 10))
        plt.imshow(maze_array, cmap=cmap, origin='upper')

        if result.ruta:
            cols = [pos[1] for pos in result.ruta]
            filas = [pos[0] for pos in result.ruta]
            plt.plot(cols, filas, color='red', linewidth=2, label='Camino')
            plt.scatter(cols[0], filas[0], color='yellow', marker='o', s=150, label='Inicio')
            plt.scatter(cols[-1], filas[-1], color='cyan', marker='x', s=150, label='Salida')

        info_text = (
            f"Tiempo de ejecución: {result.costo:.6f} s\n"
            f"Largo del camino: {result.costo}\n"
            f"Nodos explorados: {result.nodos_explorados}\n"
            f"Branching Factor Prom.: {sum(result.profundidad_ramas.values()) / len(result.profundidad_ramas) if result.profundidad_ramas else 0:.2f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.02, 0.02, info_text, fontsize=12, color="black", transform=plt.gca().transAxes,
                 verticalalignment='bottom', bbox=props)

        plt.legend(loc='upper right')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()

    def simulaciones(self, num_simulations: int = 3):
        inicio, meta = self.maze.encontrar_inicio_meta()
        if not meta:
            print("No se encontró el objetivo ('3') en el laberinto.")
            return

        algorithms = {
            "BFS": BFS(self.maze),
            "DFS": DFS(self.maze),
            "UCS": UCS(self.maze),
            "Greedy (Manhattan)": Greedy(self.maze, AlgoritmoBusqueda.heuristicaa_manhattan),
            "Greedy (Euclidiana)": Greedy(self.maze, AlgoritmoBusqueda.heuristicaa_euclidiana),
            "A* (Manhattan)": AStar(self.maze, AlgoritmoBusqueda.heuristicaa_manhattan),
            "A* (Euclidiana)": AStar(self.maze, AlgoritmoBusqueda.heuristicaa_euclidiana)
        }

        if inicio:
            print("\n--- Caso Base ---")
            for name, algorithm in algorithms.items():
                inicio_time = time.time()
                result = algorithm.search(inicio, meta)
                fin_time = time.time()
                result.costo = fin_time - inicio_time  
                self.graficar_solucion(result, f"{name} - Caso Base")

        print("\n=== Simulación con puntos de partida aleatorios ===")
        free_positions = [(i, j) for i, row in enumerate(self.maze.grid) for j, cell in enumerate(row) if cell == '0']
        random_inicios = random.sample(free_positions, min(num_simulations, len(free_positions)))

        for i, random_inicio in enumerate(random_inicios, inicio=1):
            print(f"\n--- Simulación {i} con punto de inicio: {random_inicio} ---")
            for name, algorithm in algorithms.items():
                inicio_time = time.time()
                result = algorithm.search(random_inicio, meta)
                fin_time = time.time()
                result.costo = fin_time - inicio_time
                self.graficar_solucion(result, f"{name} - Simulación {i}")
                


def generar_resultados(maze, num_simulations=3):
    inicio, meta = maze.encontrar_inicio_meta()
    if not meta:
        print("No se encontró el objetivo ('3') en el laberinto.")
        return pd.DataFrame()

    algorithms = {
        "BFS": BFS(maze),
        "DFS": DFS(maze),
        "UCS": UCS(maze),
        "Greedy (Manhattan)": Greedy(maze, AlgoritmoBusqueda.heuristicaa_manhattan),
        "Greedy (Euclidiana)": Greedy(maze, AlgoritmoBusqueda.heuristicaa_euclidiana),
        "A* (Manhattan)": AStar(maze, AlgoritmoBusqueda.heuristicaa_manhattan),
        "A* (Euclidiana)": AStar(maze, AlgoritmoBusqueda.heuristicaa_euclidiana)
    }

    results = []

    for name, algorithm in algorithms.items():
        for case, case_inicio in [("Caso Base", inicio)] + [("Aleatorio", random.choice([(i, j) for i, row in enumerate(maze.grid) for j, cell in enumerate(row) if cell == '0'])) for _ in range(num_simulations)]:
            inicio_time = time.time()
            result = algorithm.search(case_inicio, meta)
            fin_time = time.time()
            exec_time = fin_time - inicio_time
            results.append({
                "Algoritmo": name,
                "Caso": case,
                "Tiempo de ejecución (s)": round(exec_time, 6),
                "Largo del camino": result.costo if result.ruta else "N/A",
                "Nodos explorados": result.nodos_explorados,
                "Branching Factor": round(sum(result.profundidad_ramas.values()) / len(result.profundidad_ramas), 2) if result.profundidad_ramas else "N/A"
            })

    df = pd.DataFrame(results)

    df.to_csv("resultados.csv", index=False)
    print("Resultados guardados en 'resultados.csv'")

    return df

def graficar_resultados(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

    if "Tiempo de ejecución (s)" in df.columns:
        df.groupby("Algoritmo")["Tiempo de ejecución (s)"].mean().plot(
            kind="bar", ax=axes[0], title="Tiempo de Ejecución"
        )

    if "Nodos explorados" in df.columns:
        df.groupby("Algoritmo")["Nodos explorados"].mean().plot(
            kind="bar", ax=axes[1], title="Nodos Explorados"
        )

    plt.tight_layout()
    plt.show()

    


def guardar_resultados(result, algo_name, case, inicio, meta):
    csv_filename = "resultados.csv"

    new_data = {
        "Algoritmo": algo_name,
        "Caso": case,
        "Inicio": inicio,
        "Objetivo": meta,
        "Tiempo de ejecución (s)": round(result.costo, 6) if result.costo else "N/A",
        "Largo del camino": len(result.ruta) - 1 if result.ruta else "N/A",
        "Nodos explorados": result.nodos_explorados,
        "Branching Factor": round(sum(result.profundidad_ramas.values()) / len(result.profundidad_ramas), 2) if result.profundidad_ramas else "N/A"
    }
    if os.ruta.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        df = df.append(new_data, ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    df.to_csv(csv_filename, index=False)
    print(f"✅ Datos guardados en '{csv_filename}' para {algo_name} - {case}")






if __name__ == '__main__':
    maze = Maze.archivo("maze.txt")
    solver = MazeSolver(maze)
    solver.simulaciones(num_simulations=3)
