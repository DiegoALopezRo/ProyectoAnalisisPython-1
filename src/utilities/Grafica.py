import matplotlib.pyplot as plt
import numpy as np



def leer_datos(nombre_archivo):
    datos = {}
    with open(nombre_archivo) as archivo:
        conjunto_actual = []
        for linea in archivo:
            if linea.startswith("-------------------------------------------------------------------------"):
                if conjunto_actual:
                    datos[len(datos) + 1] = conjunto_actual
                conjunto_actual = []
            else:
                n, algoritmo, te = linea.strip().split(",")
                conjunto_actual.append((int(n), algoritmo, float(te)))
        if conjunto_actual:
            datos[len(datos) + 1] = conjunto_actual
    return datos


def graficar_comparativa(datos, indice_conjunto):
    conjunto = datos[indice_conjunto]
    algoritmos = sorted(set([dato[1] for dato in conjunto]))
    ns = sorted(set([dato[0] for dato in conjunto]))
    num_algoritmos = len(algoritmos)
    bar_width = 0.35
    plt.figure(figsize=(12, 6))
    index = np.arange(len(ns))
    for i, algoritmo in enumerate(algoritmos):
        tiempos = [dato[2] for dato in conjunto if dato[1] == algoritmo]
        plt.bar(index + i * bar_width, tiempos, bar_width, label=algoritmo)
    plt.xlabel("Algoritmo")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title(f"Matriz Tamaño {2**indice_conjunto}x{2**indice_conjunto}")
    plt.xticks(index + bar_width * (num_algoritmos - 1) / 2, ns)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./src/matrixes_files/img/comparativa_matriz{2**indice_conjunto}x{2**indice_conjunto}.png")
    #plt.show()

