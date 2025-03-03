import tkinter as tk
from tkinter import ttk
import threading
import os
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from maze import Maze, MazeSolver, BFS, DFS, UCS, Greedy, AStar, AlgoritmoBusqueda

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Algoritmos de Búsqueda")

        self.label_maze = tk.Label(root, text="Selecciona un laberinto:")
        self.label_maze.pack()

        self.maze_files = [f for f in os.listdir() if f.endswith(".txt")]
        self.selected_maze = tk.StringVar()
        self.combo_maze = ttk.Combobox(root, textvariable=self.selected_maze, values=self.maze_files, state="readonly")
        self.combo_maze.pack()
        
        self.simulacion_viva = tk.BooleanVar(value=False)  
        self.checkbox_simulacion = tk.Checkbutton(root, text="Simulación en vivo", variable=self.simulacion_viva)
        self.checkbox_simulacion.pack(pady=5)


        self.label_algoritmos = tk.Label(root, text="Selecciona un algoritmo:")
        self.label_algoritmos.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

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
        
        self.tree.heading("Algoritmo", text="Algoritmo")
        self.tree.heading("Caso", text="Caso")
        self.tree.heading("Inicio", text="Inicio")
        self.tree.heading("Objetivo", text="Objetivo")
        self.tree.heading("Tiempo", text="Tiempo (s)")
        self.tree.heading("Largo", text="Largo Camino")
        self.tree.heading("Nodos", text="Nodos Explorados")
        self.tree.heading("Factor", text="Branching Factor")

        self.tree.pack(expand=True, fill="both")

        self.plot_button = tk.Button(root, text="Ver Gráficos", command=self.resultados)
        self.plot_button.pack(pady=10)

    def ejecutar_simulacion(self, algo_name, algo_class):
        maze_path = self.selected_maze.get()
        if not maze_path:
            print("Selecciona un laberinto antes de iniciar la simulación.")
            return
        
        print(f"Cargando laberinto desde: {maze_path}")  

        maze = Maze.archivo(maze_path)
        start, goal = maze.encontrar_inicio_meta()

        if not goal:
            print("No se encontró un objetivo ('3') en el laberinto.")
            return
        
        print(f"Algoritmo seleccionado: {algo_name}") 

        algoritmo = algo_class(maze)

        if start:
            print("Ejecutando caso base...")
            self.root.after(0, self.ejecutar_grafica, algoritmo, start, goal, f"{algo_name} - Caso Base")

        free_positions = [(i, j) for i, row in enumerate(maze.grid) for j, cell in enumerate(row) if cell == '0']
        if free_positions:
            random_start = random.choice(free_positions)
            print(f"Ejecutando caso aleatorio desde {random_start}...")
            self.root.after(0, self.ejecutar_grafica, algoritmo, random_start, goal, f"{algo_name} - Caso Aleatorio")

    def actualizar_tabla(self, algo_name, case, start, goal, result):
        self.tree.insert("", "end", values=(
            algo_name,
            case,
            start,
            goal,
            round(result.costo, 6) if result.costo else "N/A",
            len(result.ruta) - 1 if result.ruta else "N/A",
            result.nodos_explorados,
            round(sum(result.profundidad_ramas.values()) / len(result.profundidad_ramas), 2) if result.profundidad_ramas else "N/A"
        ))

    def ejecutar_grafica(self, algoritmo, start, goal, title):
        inicio_time = time.time()
        result = algoritmo.busqueda(start, goal)
        fin_time = time.time()
        exec_time = fin_time - inicio_time
        result.costo = exec_time 

        self.actualizar_tabla(algoritmo.__class__.__name__, title, start, goal, result)

        solver = MazeSolver(algoritmo.maze)
        solver.graficar_solucion(result, title)

    def resultados(self):
        data = []
        for row in self.tree.get_children():
            values = self.tree.item(row)["values"]
            if values:  
                try:
                    data.append({
                        "Algoritmo": values[0],
                        "Tiempo (s)": float(values[4]) if values[4] != "N/A" else None,
                        "Largo Camino": int(values[5]) if values[5] != "N/A" else None,
                        "Nodos Explorados": int(values[6]),
                        "Branching Factor": float(values[7]) if values[7] != "N/A" else None
                    })
                except ValueError:
                    continue  

        df = pd.DataFrame(data)

        if df.empty:
            print("No hay datos en la tabla para graficar.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        df.groupby("Algoritmo")["Tiempo (s)"].mean().plot(kind="bar", ax=axes[0, 0], title="Tiempo de Ejecución")

        df.dropna(subset=["Largo Camino"]).groupby("Algoritmo")["Largo Camino"].mean().plot(kind="bar", ax=axes[0, 1], title="Largo del Camino")

        df.groupby("Algoritmo")["Nodos Explorados"].mean().plot(kind="bar", ax=axes[1, 0], title="Nodos Explorados")

        df.dropna(subset=["Branching Factor"]).groupby("Algoritmo")["Branching Factor"].mean().plot(kind="bar", ax=axes[1, 1], title="Branching Factor")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
