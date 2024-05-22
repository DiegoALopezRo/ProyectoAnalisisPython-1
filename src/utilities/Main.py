
from Metodo import *
from Grafica import *

@staticmethod
def main():

    n = 11
    for i in range(1,n+1):
        ejecutarMetodos(2**i)

    # Nombre del archivo
    nombre_archivo = f"./src/matrixes_files/resultados.txt"

    # Leer los datos
    datos = leer_datos(nombre_archivo)

    # Generar y guardar la comparativa en barras para cada conjunto de datos
    for indice_conjunto in datos:
        graficar_comparativa(datos, indice_conjunto)
    

# Para probar el método main
if __name__ == "__main__":
    main()

        