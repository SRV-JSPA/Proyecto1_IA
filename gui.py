import tkinter as tk
from tkinter import ttk
import threading
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import pandas as pd
from maze import Maze, MazeSolver, BFS, DFS, UCS, Greedy, AStar, AlgoritmoBusqueda

class SimulacionVivo:
    def __init__(self, maze, algorithm):
        self.maze = maze
        self.algorithm = algorithm
        self.solution_steps = []  
        self.generador_pasos()

    def generador_pasos(self):
        start, goal = self.maze.encontrar_inicio_meta()
        if not start or not goal:
            print("No se encontró el inicio o la meta.")
            return

        queue = [(start, [start])]
        visited = set([start])

        while queue:
            current, path = queue.pop(0)
            self.solution_steps.append(list(path))

            if current == goal:
                break

            for neighbor in self.maze.obtener_vecinos(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    def animacion(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        mapping = {'0': 0, '1': 1, '2': 2, '3': 3}
        maze_array = np.array([[mapping.get(cell, 0) for cell in row] for row in self.maze.grid])
        cmap = ListedColormap(['white', 'black', 'green', 'blue'])

        ax.imshow(maze_array, cmap=cmap, origin='upper')
        path_plot, = ax.plot([], [], 'r-', linewidth=2, label="Camino")

        def update(frame):
            if frame >= len(self.solution_steps):
                return path_plot,
            path = self.solution_steps[frame]
            cols = [p[1] for p in path]
            rows = [p[0] for p in path]
            path_plot.set_data(cols, rows)
            return path_plot,

        ani = FuncAnimation(fig, update, frames=len(self.solution_steps), interval=100, repeat=False)
        plt.legend()
        plt.show()

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Algoritmos de Búsqueda")

        self.simulacion_viva = tk.BooleanVar()
        
        self.label_maze = tk.Label(root, text="Selecciona un laberinto:")
        self.label_maze.pack()
        
        self.maze_files = [f for f in os.listdir() if f.endswith(".txt")]
        self.selected_maze = tk.StringVar()
        self.combo_maze = ttk.Combobox(root, textvariable=self.selected_maze, values=self.maze_files, state="readonly")
        self.combo_maze.pack()

        self.label_algoritmos = tk.Label(root, text="Selecciona un algoritmo:")
        self.label_algoritmos.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

        tk.Checkbutton(root, text="Simulación en vivo", variable=self.simulacion_viva).pack()

        algoritmos = {
            "BFS": BFS,
            "DFS": DFS,
            "UCS": UCS,
            "Greedy (Manhattan)": lambda maze: Greedy(maze, AlgoritmoBusqueda.heuristicaa_manhattan),
            "Greedy (Euclidiana)": lambda maze: Greedy(maze, AlgoritmoBusqueda.heuristicaa_euclidiana),
            "A* (Manhattan)": lambda maze: AStar(maze, AlgoritmoBusqueda.heuristicaa_manhattan),
            "A* (Euclidiana)": lambda maze: AStar(maze, AlgoritmoBusqueda.heuristicaa_euclidiana)
        }

        for name, algo_class in algoritmos.items():
            btn = tk.Button(self.buttons_frame, text=name, command=lambda n=name, a=algo_class: self.ejecutar_simulacion(n, a))
            btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=10)

        self.tree = ttk.Treeview(self.results_frame, columns=("Algoritmo", "Caso", "Inicio", "Objetivo", "Tiempo", "Largo", "Nodos", "Factor"), show="headings")
        for col in ("Algoritmo", "Caso", "Inicio", "Objetivo", "Tiempo", "Largo", "Nodos", "Factor"):
            self.tree.heading(col, text=col)
        self.tree.pack(expand=True, fill="both")

        self.plot_button = tk.Button(root, text="Ver Gráficos", command=self.resultados)
        self.plot_button.pack(pady=10)


    def ejecutar_simulacion(self, algo_name, algo_class):
        maze_path = self.selected_maze.get()
        if not maze_path:
            print("Selecciona un laberinto antes de iniciar la simulación.")
            return

        maze = Maze.archivo(maze_path)
        start, goal = maze.encontrar_inicio_meta()

        if not goal:
            print("No se encontró un objetivo ('3') en el laberinto.")
            return

        if self.simulacion_viva.get():
            solver = SimulacionVivo(maze, algo_class(maze))
            solver.animacion()
        else:
            algoritmo = algo_class(maze)
            inicio_time = time.time()
            result = algoritmo.busqueda(start, goal)
            fin_time = time.time()
            result.costo = fin_time - inicio_time

            self.actualizar_tabla(algo_name, "Caso Base", start, goal, result)
            solver = MazeSolver(algoritmo.maze)
            solver.graficar_solucion(result, f"{algo_name} - Caso Base")

            free_positions = [(i, j) for i, row in enumerate(maze.grid) for j, cell in enumerate(row) if cell == '0']
            if free_positions:
                random_start = random.choice(free_positions)
                print(f"Ejecutando caso aleatorio desde {random_start}...")
                
                inicio_time = time.time()
                result = algoritmo.busqueda(random_start, goal)
                fin_time = time.time()
                result.costo = fin_time - inicio_time

                self.actualizar_tabla(algo_name, "Caso Aleatorio", random_start, goal, result)
                solver.graficar_solucion(result, f"{algo_name} - Caso Aleatorio")

    def actualizar_tabla(self, algo_name, case, start, goal, result):
        self.tree.insert("", "end", values=(
            algo_name, case, start, goal,
            round(result.costo, 6) if result.costo else "N/A",
            len(result.ruta) - 1 if result.ruta else "N/A",
            result.nodos_explorados,
            round(sum(result.profundidad_ramas.values()) / len(result.profundidad_ramas), 2) if result.profundidad_ramas else "N/A"
        ))
        
    def resultados(self):
        data = []
        for row in self.tree.get_children():
            values = self.tree.item(row)["values"]
            if values:  
                try:
                    data.append({
                        "Algoritmo": values[0],
                        "Tiempo (s)": float(values[4]) if values[4] != "N/A" else None,
                        "Nodos Explorados": int(values[6])
                    })
                except ValueError:
                    continue  

        df = pd.DataFrame(data)

        if df.empty:
            print("No hay datos en la tabla para graficar.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

        df.groupby("Algoritmo")["Tiempo (s)"].mean().plot(kind="bar", ax=axes[0], title="Tiempo de Ejecución")
        df.groupby("Algoritmo")["Nodos Explorados"].mean().plot(kind="bar", ax=axes[1], title="Nodos Explorados")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
