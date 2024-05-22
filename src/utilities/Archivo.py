import random


def leer_archivo_matriz(ruta):
    matriz = None
    try:
        with open(ruta, 'r') as file:
            # Primera linea nos dice longitud de la matriz
            linea = file.readline()
            longitud = int(linea.strip())
            matriz = [[0] * longitud for _ in range(longitud)]
            # Las siguientes lineas son filas de la matriz
            fila = 0  # Para recorrer las filas de la matriz
            linea = file.readline().strip()
            while linea:
                """
                Tenemos todos los enteros JUNTOS en el String linea. Con
                split() los SEPARAMOS en un array donde cada entero es un
                String individual. Con un bucle, los parseamos a Integer para
                guardarlos en la matriz
                """
                enteros = linea.split(" ")
                for i in range(len(enteros)):
                    matriz[fila][i] = int(enteros[i])
                fila += 1  # Incrementamos fila para la próxima línea de enteros
                linea = file.readline().strip()  # Leemos siguiente línea
    except FileNotFoundError:
        print("No se encuentra archivo")
    except ValueError:
        print("No se pudo convertir a entero")
    except IOError:
        print("Error accediendo al archivo.")

    return matriz

def generar_txt_matriz_prueba(n):
    matriz = [[0] * n for _ in range(n)]
    ruta_nuevo = f"./src/matrixes_files/matriz{n}x{n}.txt"

    try:
        with open(ruta_nuevo, 'w') as file:
            file.write(f"{n}\n")
            for i in range(n):
                for j in range(n):
                    matriz[i][j] = random.randint(100000, 999999)
                    if j < n - 1:
                        file.write(f"{matriz[i][j]} ")
                    else:
                        file.write(f"{matriz[i][j]}")
                file.write("\n")
    except IOError as e:
        print("Error al escribir en el archivo:", e)

def guardar_resultado(n, te, algoritmo):
    # Formar la cadena a guardar en el archivo
    linea = f"{n},{algoritmo},{te}\n"
    
    # Nombre del archivo
    nombre_archivo = f"./src/matrixes_files/resultados.txt"
    
    if n == 0:
        # Abrir el archivo en modo escritura (si no existe se crea)
        with open(nombre_archivo, "a") as archivo:
            # Escribir la línea en el archivo
            archivo.write("-------------------------------------------------------------------------\n")
    else:
        # Abrir el archivo en modo escritura (si no existe se crea)
        with open(nombre_archivo, "a") as archivo:
            # Escribir la línea en el archivo
            archivo.write(linea)