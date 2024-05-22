import timeit
import math
import numpy as np

from Archivo import * 
from concurrent.futures import ThreadPoolExecutor

SEQUENTIAL_THRESHOLD = 128 #Cantidad de hilos que se pueden ejecutar al tiempo, empezar desde 64 (2^n)

       

def ejecutarMetodos(n):
    ruta = "./src/matrixes_files/matriz{}x{}.txt".format(n, n)

    # Generar una matriz de prueba de 4x4
    generar_txt_matriz_prueba(n)
    matriz_a = leer_archivo_matriz(ruta)
    generar_txt_matriz_prueba(n)
    matriz_b = leer_archivo_matriz(ruta)
    
    # Imprimir las matrices generadas
    #print("Matriz A:")
    #imprimir_matriz(matriz_a)

    #print("Matriz B:")
    #imprimir_matriz(matriz_b)

    # Crear una matriz vacía para almacenar el resultado de la multiplicación
    matriz_c = [[0 for _ in range(n)] for _ in range(n)]

    n = len(matriz_a)  # Número de filas en matrizA
    p = len(matriz_b)  # Número de columnas en matrizA / Número de filas en matrizB
    m = len(matriz_b[0])  # Número de columnas en matrizB

    print(n, p, m)
    
    # Ejecutar el algoritmo de multiplicación de matrice
    # Llamadas para todos los métodos

    ejecutar_metodo(NaivOnArray, "NaivOnArray", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(NaivLoopUnrollingTwo, "NaivLoopUnrollingTwo", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(NaivLoopUnrollingFour, "NaivLoopUnrollingFour", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(WinogradOriginal, "WinogradOriginal", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(WinogradScaled, "WinogradScaled", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(StrassenNaiv, "StrassenNaiv", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(StrassenWinograd, "StrassenWinograd", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(III3SequentialBlock, "III3SequentialBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(III4ParallelBlock, "III4ParallelBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo2(III5EnhancedParallelBlock, "III5EnhancedParallelBlock", matriz_a, matriz_b, n)
    ejecutar_metodo(IV3SequentialBlock, "IV3SequentialBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(IV4ParallelBlock, "IV4ParallelBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(V3SequentialBlock, "V3SequentialBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo(V4ParallelBlock, "V4ParallelBlock", matriz_a, matriz_b, matriz_c, n, p, m)
    ejecutar_metodo2(IV5EnhancedParallelBlock, "IV5EnhancedParallelBlock", matriz_a, matriz_b, n)

    guardar_resultado(0, 0, "")

@staticmethod
def ejecutar_metodo(metodo, nombre, matriz_a, matriz_b, matriz_c, n, p, m):
    tiempo_promedio = timeit.timeit(lambda: metodo(matriz_a, matriz_b, matriz_c, n, p, m), number=1)
    print(f"Tiempo promedio transcurrido en {nombre}: {tiempo_promedio} segundos")
    guardar_resultado(n, tiempo_promedio, nombre)

@staticmethod
def ejecutar_metodo2(metodo, nombre, matriz_a, matriz_b, n):
    tiempo_promedio = timeit.timeit(lambda: metodo(matriz_a, matriz_b, n, SEQUENTIAL_THRESHOLD), number=1)
    print(f"Tiempo promedio transcurrido en {nombre}: {tiempo_promedio} segundos")
    guardar_resultado(n, tiempo_promedio, nombre)

def imprimir_matriz(matriz):
    # Mostramos la matriz leída
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            print(matriz[i][j], end=" ")
        print()
    print("\n")

def NaivOnArray(matrizA, matrizB, matrizC, N, P, M):
    """
    Implementación del algoritmo NaiveOnArray para la multiplicación de matrices.

    :param matrizA: La matriz A.
    :param matrizB: La matriz B.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A y en la matriz C.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de columnas en la matriz B y en la matriz C.
    """
    # Itera sobre las filas de la matriz A
    for i in range(N):
        # Itera sobre las columnas de la matriz B
        for j in range(M):
            # Inicializa el valor de la celda (i, j) en la matriz C como 0
            matrizC[i][j] = 0
            # Itera sobre las columnas de la matriz A o las filas de la matriz B
            for k in range(P):
                # Realiza la multiplicación de matrices y acumula el resultado en la matriz C
                matrizC[i][j] += matrizA[i][k] * matrizB[k][j]

def NaivLoopUnrollingTwo(matrizA, matrizB, matrizC, N, P, M):
    """
    Implementación del algoritmo NaivLoopUnrollingTwo para la multiplicación de matrices.

    :param matrizA: La matriz A.
    :param matrizB: La matriz B.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A y en la matriz C.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de columnas en la matriz B y en la matriz C.
    """
    for i in range(N):
        for j in range(M):
            aux = 0
            if P % 2 == 0:
                for k in range(0, P, 2):
                    aux += matrizA[i][k] * matrizB[k][j] + matrizA[i][k + 1] * matrizB[k + 1][j]
            else:
                PP = P - 1
                for k in range(0, PP, 2):
                    aux += matrizA[i][k] * matrizB[k][j] + matrizA[i][k + 1] * matrizB[k + 1][j]
                aux += matrizA[i][PP] * matrizB[PP][j]
            matrizC[i][j] = aux

def NaivLoopUnrollingFour(A, B, Result, N, P, M):
    """
    Implementación del algoritmo NaivLoopUnrollingFour para la multiplicación de matrices.

    :param A: La matriz A.
    :param B: La matriz B.
    :param Result: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A y en la matriz C.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de columnas en la matriz B y en la matriz C.
    """
    for i in range(N):
        for j in range(M):
            aux = 0
            if P % 4 == 0:
                for k in range(0, P, 4):
                    aux += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j]
            elif P % 4 == 1:
                PP = P - 1
                for k in range(0, PP, 4):
                    aux += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j]
                aux += A[i][PP] * B[PP][j]
            elif P % 4 == 2:
                PP = P - 2
                PPP = P - 1
                for k in range(0, PP, 4):
                    aux += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j]
                aux += A[i][PP] * B[PP][j] + A[i][PPP] * B[PPP][j]
            else:
                PP = P - 3
                PPP = P - 2
                PPPP = P - 1
                for k in range(0, PP, 4):
                    aux += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] + A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j]
                aux += A[i][PP] * B[PP][j] + A[i][PPP] * B[PPP][j] + A[i][PPPP] * B[PPPP][j]
            Result[i][j] = aux

def WinogradOriginal(A, B, Result, N, P, M):
    """
    Implementación del algoritmo WinogradOriginal para la multiplicación de matrices.

    :param A: La matriz A.
    :param B: La matriz B.
    :param Result: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A y el número de columnas en la matriz B.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de filas en la matriz A y en la matriz C.
    """
    upsilon = P % 2
    gamma = P - upsilon
    y = [0] * M
    z = [0] * N

    for i in range(M):
        aux = 0
        for j in range(0, gamma, 2):
            aux += A[i][j] * A[i][j+1]
        y[i] = aux

    for i in range(N):
        aux = 0
        for j in range(0, gamma, 2):
            aux += B[j][i] * B[j+1][i]
        z[i] = aux

    if upsilon == 1:
        PP = P - 1
        for i in range(M):
            for k in range(N):
                aux = 0
                for j in range(0, gamma, 2):
                    aux += (A[i][j] + B[j+1][k]) * (A[i][j+1] + B[j][k])
                Result[i][k] = aux - y[i] - z[k] + A[i][PP] * B[PP][k]
    else:
        for i in range(M):
            for k in range(N):
                aux = 0
                for j in range(0, gamma, 2):
                    aux += (A[i][j] + B[j+1][k]) * (A[i][j+1] + B[j][k])
                Result[i][k] = aux - y[i] - z[k]

    # Liberar memoria
    y = None
    z = None

def WinogradScaled(A, B, Result, N, P, M):
    """
    Implementación del algoritmo WinogradScaled para la multiplicación de matrices.

    :param A: La matriz A.
    :param B: La matriz B.
    :param Result: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de columnas en la matriz B.
    """
    # Crear copias escaladas de A y B
    CopyA = [[0] * P for _ in range(N)]
    CopyB = [[0] * M for _ in range(P)]
    
    # Factores de escala
    a = NormInf(A, N, P)
    b = NormInf(B, P, M)
    lambda_val = math.floor(0.5 + math.log(b/a) / math.log(4))
    
    # Escalar
    MultiplyWithScalar(A, CopyA, N, P, 2 ** lambda_val)
    MultiplyWithScalar(B, CopyB, P, M, 2 ** -lambda_val)
    
    # Utilizar Winograd con las matrices escaladas
    WinogradOriginal(CopyA, CopyB, Result, N, P, M)

def MultiplyWithScalar(A, B, N, M, scalar):
    """
    Multiplica una matriz por un escalar.

    :param A: La matriz original.
    :param B: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz.
    :param M: El número de columnas en la matriz.
    :param scalar: El escalar por el que se multiplicará la matriz.
    """
    for i in range(N):
        for j in range(M):
            B[i][j] = A[i][j] * scalar

def NormInf(A, N, M):
    """
    Calcula la norma infinito de una matriz.

    :param A: La matriz.
    :param N: El número de filas en la matriz.
    :param M: El número de columnas en la matriz.
    :return: La norma infinito de la matriz.
    """
    max_sum = float('-inf')
    for i in range(N):
        row_sum = sum(abs(A[i][j]) for j in range(M))
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

def StrassenNaiv(matrizA, matrizB, matrizC, N, P, M):
    """
    Implementación del algoritmo StrassenNaiv para la multiplicación de matrices.

    :param matrizA: La matriz A.
    :param matrizB: La matriz B.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas en la matriz A.
    :param P: El número de columnas en la matriz A y el número de filas en la matriz B.
    :param M: El número de columnas en la matriz B.
    """
    MaxSize = max(N, P)
    if MaxSize < 16:
        MaxSize = 16  # De lo contrario, no es posible calcular k
    k = int(math.floor(math.log(MaxSize) / math.log(2)) - 4)
    m = int(math.floor(MaxSize * (2 ** -k)) + 1)
    NewSize = m * (2 ** k)

    # Agregar filas y columnas de ceros para usar el algoritmo de Strassen
    NewA = [[0] * NewSize for _ in range(NewSize)]
    NewB = [[0] * NewSize for _ in range(NewSize)]
    AuxResult = [[0] * NewSize for _ in range(NewSize)]

    for i in range(N):
        for j in range(P):
            NewA[i][j] = matrizA[i][j]

    for i in range(P):
        for j in range(M):
            NewB[i][j] = matrizB[i][j]

    StrassenNaivStep(NewA, NewB, AuxResult, NewSize, m)

    # Extraer el resultado
    for i in range(N):
        for j in range(M):
            matrizC[i][j] = AuxResult[i][j]

def max(N, P):
    """
    Calcula el máximo entre dos números.

    :param N: El primer número.
    :param P: El segundo número.
    :return: El máximo entre N y P.
    """
    return max(P, N) if N < P else N

def Minus(A, B, Result, N, M):
    """
    Resta de matrices.

    :param A: La primera matriz.
    :param B: La segunda matriz.
    :param Result: La matriz donde se almacenará el resultado de la resta.
    :param N: El número de filas en las matrices.
    :param M: El número de columnas en las matrices.
    """
    for i in range(N):
        for j in range(M):
            Result[i][j] = A[i][j] - B[i][j]

def Plus(A, B, Result, N, M):
    """
    Suma de matrices.

    :param A: La primera matriz.
    :param B: La segunda matriz.
    :param Result: La matriz donde se almacenará el resultado de la suma.
    :param N: El número de filas en las matrices.
    :param M: El número de columnas en las matrices.
    """
    for i in range(N):
        for j in range(M):
            Result[i][j] = A[i][j] + B[i][j]

def StrassenNaivStep(A, B, Result, N, m):
    """
    Implementación del paso del algoritmo StrassenNaiv para la multiplicación de matrices.

    :param A: La matriz A.
    :param B: La matriz B.
    :param Result: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El tamaño de las matrices (N x N).
    :param m: Un parámetro m utilizado para determinar el tamaño de la división de la matriz.
    """
    if N % 2 == 0 and N > m:
        NewSize = N // 2

        # Descomponer A y B
        A11 = [[0] * NewSize for _ in range(NewSize)]
        A12 = [[0] * NewSize for _ in range(NewSize)]
        A21 = [[0] * NewSize for _ in range(NewSize)]
        A22 = [[0] * NewSize for _ in range(NewSize)]
        B11 = [[0] * NewSize for _ in range(NewSize)]
        B12 = [[0] * NewSize for _ in range(NewSize)]
        B21 = [[0] * NewSize for _ in range(NewSize)]
        B22 = [[0] * NewSize for _ in range(NewSize)]

        ResultPart11 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart12 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart21 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart22 = [[0] * NewSize for _ in range(NewSize)]

        Helper1 = [[0] * NewSize for _ in range(NewSize)]
        Helper2 = [[0] * NewSize for _ in range(NewSize)]

        Aux1 = [[0] * NewSize for _ in range(NewSize)]
        Aux2 = [[0] * NewSize for _ in range(NewSize)]
        Aux3 = [[0] * NewSize for _ in range(NewSize)]
        Aux4 = [[0] * NewSize for _ in range(NewSize)]
        Aux5 = [[0] * NewSize for _ in range(NewSize)]
        Aux6 = [[0] * NewSize for _ in range(NewSize)]
        Aux7 = [[0] * NewSize for _ in range(NewSize)]

        for i in range(NewSize):
            for j in range(NewSize):
                A11[i][j] = A[i][j]
                A12[i][j] = A[i][NewSize + j]
                A21[i][j] = A[NewSize + i][j]
                A22[i][j] = A[NewSize + i][NewSize + j]
                B11[i][j] = B[i][j]
                B12[i][j] = B[i][NewSize + j]
                B21[i][j] = B[NewSize + i][j]
                B22[i][j] = B[NewSize + i][NewSize + j]

        # Computar las siete variables auxiliares
        Plus(A11, A22, Helper1, NewSize, NewSize)
        Plus(B11, B22, Helper2, NewSize, NewSize)
        StrassenNaivStep(Helper1, Helper2, Aux1, NewSize, m)

        Plus(A21, A22, Helper1, NewSize, NewSize)
        StrassenNaivStep(Helper1, B11, Aux2, NewSize, m)

        Minus(B12, B22, Helper1, NewSize, NewSize)
        StrassenNaivStep(A11, Helper1, Aux3, NewSize, m)

        Minus(B21, B11, Helper1, NewSize, NewSize)
        StrassenNaivStep(A22, Helper1, Aux4, NewSize, m)

        Plus(A11, A12, Helper1, NewSize, NewSize)
        StrassenNaivStep(Helper1, B22, Aux5, NewSize, m)

        Minus(A21, A11, Helper1, NewSize, NewSize)
        Plus(B11, B12, Helper2, NewSize, NewSize)
        StrassenNaivStep(Helper1, Helper2, Aux6, NewSize, m)

        Minus(A12, A22, Helper1, NewSize, NewSize)
        Plus(B21, B22, Helper2, NewSize, NewSize)
        StrassenNaivStep(Helper1, Helper2, Aux7, NewSize, m)

        # Computar las cuatro partes del resultado
        Plus(Aux1, Aux4, ResultPart11, NewSize, NewSize)
        Minus(ResultPart11, Aux5, ResultPart11, NewSize, NewSize)
        Plus(ResultPart11, Aux7, ResultPart11, NewSize, NewSize)

        Plus(Aux3, Aux5, ResultPart12, NewSize, NewSize)

        Plus(Aux2, Aux4, ResultPart21, NewSize, NewSize)

        Plus(Aux1, Aux3, ResultPart22, NewSize, NewSize)
        Minus(ResultPart22, Aux2, ResultPart22, NewSize, NewSize)
        Plus(ResultPart22, Aux6, ResultPart22, NewSize, NewSize)

        # Almacenar los resultados en la matriz de resultados
        for i in range(NewSize):
            for j in range(NewSize):
                Result[i][j] = ResultPart11[i][j]
                Result[i][NewSize + j] = ResultPart12[i][j]
                Result[NewSize + i][j] = ResultPart21[i][j]
                Result[NewSize + i][NewSize + j] = ResultPart22[i][j]

        # Liberar memoria
        A11 = A12 = A21 = A22 = None
        B11 = B12 = B21 = B22 = None
        ResultPart11 = ResultPart12 = ResultPart21 = ResultPart22 = None
        Helper1 = Helper2 = None
        Aux1 = Aux2 = Aux3 = Aux4 = Aux5 = Aux6 = Aux7 = None

    else:
        # Usar el algoritmo naïve
        NaivStandard(A, B, Result, N, N, N)

def NaivStandard(matrizA, matrizB, matrizC, N, P, M):
    """
    Implementación del algoritmo estándar para la multiplicación de matrices.

    :param matrizA: La matriz A.
    :param matrizB: La matriz B.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas de la matriz A.
    :param P: El número de columnas de la matriz A (y el número de filas de la matriz B).
    :param M: El número de columnas de la matriz B.
    """
    for i in range(N):
        for j in range(M):
            aux = 0
            for k in range(P):
                aux += matrizA[i][k] * matrizB[k][j]
            matrizC[i][j] = aux

def StrassenWinograd(matrizA, matrizB, matrizC, N, P, M):
    """
    Implementación del algoritmo Strassen-Winograd para la multiplicación de matrices.

    :param matrizA: La matriz A.
    :param matrizB: La matriz B.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El número de filas de la matriz A.
    :param P: El número de columnas de la matriz A (y el número de filas de la matriz B).
    :param M: El número de columnas de la matriz B.
    """
    # Calcular el tamaño máximo necesario
    MaxSize = max(N, P)
    if MaxSize < 16:
        MaxSize = 16  # De lo contrario, no es posible calcular k
    k = int((MaxSize).bit_length() - 4)
    m = int(MaxSize * 2 ** -k) + 1
    NewSize = m * 2 ** k

    # Agregar filas y columnas de ceros para usar el algoritmo Strassen
    NewA = [[0] * NewSize for _ in range(NewSize)]
    NewB = [[0] * NewSize for _ in range(NewSize)]
    AuxResult = [[0] * NewSize for _ in range(NewSize)]

    # Llenar las matrices extendidas con los valores de las matrices originales
    for i in range(N):
        for j in range(P):
            NewA[i][j] = matrizA[i][j]

    for i in range(P):
        for j in range(M):
            NewB[i][j] = matrizB[i][j]

    StrassenWinogradStep(NewA, NewB, AuxResult, NewSize, m)

    # Extraer el resultado
    for i in range(N):
        for j in range(M):
            matrizC[i][j] = AuxResult[i][j]

def StrassenWinogradStep(A, B, Result, N, m):
    """
    Implementación del paso del algoritmo Strassen-Winograd para la multiplicación de matrices.

    :param A: La matriz A.
    :param B: La matriz B.
    :param Result: La matriz donde se almacenará el resultado de la multiplicación.
    :param N: El tamaño de las matrices A y B.
    :param m: Un parámetro de optimización.
    """
    if N % 2 == 0 and N > m:  # Uso recursivo de StrassenNaivStep
        NewSize = N // 2

        # Descomponer A y B
        # Crear ResultPart, Aux1,...,Aux7 y Helper1, Helper2
        A1 = [[0] * NewSize for _ in range(NewSize)]
        A2 = [[0] * NewSize for _ in range(NewSize)]
        B1 = [[0] * NewSize for _ in range(NewSize)]
        B2 = [[0] * NewSize for _ in range(NewSize)]

        A11 = [[0] * NewSize for _ in range(NewSize)]
        A12 = [[0] * NewSize for _ in range(NewSize)]
        A21 = [[0] * NewSize for _ in range(NewSize)]
        A22 = [[0] * NewSize for _ in range(NewSize)]
        B11 = [[0] * NewSize for _ in range(NewSize)]
        B12 = [[0] * NewSize for _ in range(NewSize)]
        B21 = [[0] * NewSize for _ in range(NewSize)]
        B22 = [[0] * NewSize for _ in range(NewSize)]

        ResultPart11 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart12 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart21 = [[0] * NewSize for _ in range(NewSize)]
        ResultPart22 = [[0] * NewSize for _ in range(NewSize)]

        Helper1 = [[0] * NewSize for _ in range(NewSize)]
        Helper2 = [[0] * NewSize for _ in range(NewSize)]

        Aux1 = [[0] * NewSize for _ in range(NewSize)]
        Aux2 = [[0] * NewSize for _ in range(NewSize)]
        Aux3 = [[0] * NewSize for _ in range(NewSize)]
        Aux4 = [[0] * NewSize for _ in range(NewSize)]
        Aux5 = [[0] * NewSize for _ in range(NewSize)]
        Aux6 = [[0] * NewSize for _ in range(NewSize)]
        Aux7 = [[0] * NewSize for _ in range(NewSize)]
        Aux8 = [[0] * NewSize for _ in range(NewSize)]
        Aux9 = [[0] * NewSize for _ in range(NewSize)]

        # Llenar las nuevas matrices
        for i in range(NewSize):
            for j in range(NewSize):
                A11[i][j] = A[i][j]
                A12[i][j] = A[i][NewSize + j]
                A21[i][j] = A[NewSize + i][j]
                A22[i][j] = A[NewSize + i][NewSize + j]
                B11[i][j] = B[i][j]
                B12[i][j] = B[i][NewSize + j]
                B21[i][j] = B[NewSize + i][j]
                B22[i][j] = B[NewSize + i][NewSize + j]

        Minus(A11, A21, A1, NewSize, NewSize)
        Minus(A22, A1, A2, NewSize, NewSize)
        Minus(B22, B12, B1, NewSize, NewSize)
        Plus(B1, B11, B2, NewSize, NewSize)

        StrassenWinogradStep(A11, B11, Aux1, NewSize, m)
        StrassenWinogradStep(A12, B21, Aux2, NewSize, m)
        StrassenWinogradStep(A2, B2, Aux3, NewSize, m)
        Plus(A21, A22, Helper1, NewSize, NewSize)
        Minus(B12, B11, Helper2, NewSize, NewSize)
        StrassenWinogradStep(Helper1, Helper2, Aux4, NewSize, m)
        StrassenWinogradStep(A1, B1, Aux5, NewSize, m)
        Minus(A12, A2, Helper1, NewSize, NewSize)
        StrassenWinogradStep(Helper1, B22, Aux6, NewSize, m)
        Minus(B21, B2, Helper1, NewSize, NewSize)
        StrassenWinogradStep(A22, Helper1, Aux7, NewSize, m)
        Plus(Aux1, Aux3, Aux8, NewSize, NewSize)
        Plus(Aux8, Aux4, Aux9, NewSize, NewSize)

        # Calcular las cuatro partes del resultado
        Plus(Aux1, Aux2, ResultPart11, NewSize, NewSize)
        Plus(Aux9, Aux6, ResultPart12, NewSize, NewSize)
        Plus(Aux8, Aux5, Helper1, NewSize, NewSize)
        Plus(Helper1, Aux7, ResultPart21, NewSize, NewSize)
        Plus(Aux9, Aux5, ResultPart22, NewSize, NewSize)

        # Almacenar los resultados en la matriz de resultado
        for i in range(NewSize):
            for j in range(NewSize):
                Result[i][j] = ResultPart11[i][j]
                Result[i][NewSize + j] = ResultPart12[i][j]
                Result[NewSize + i][j] = ResultPart21[i][j]
                Result[NewSize + i][NewSize + j] = ResultPart22[i][j]

        # Liberar las variables auxiliares
        A1 = None
        A2 = None
        B1 = None
        B2 = None
        A11 = None
        A12 = None
        A21 = None
        A22 = None
        B11 = None
        B12 = None
        B21 = None
        B22 = None
        ResultPart11 = None
        ResultPart12 = None
        ResultPart21 = None
        ResultPart22 = None
        Helper1 = None
        Helper2 = None
        Aux1 = None
        Aux2 = None
        Aux3 = None
        Aux4 = None
        Aux5 = None
        Aux6 = None
        Aux7 = None
        Aux8 = None
        Aux9 = None
    else:
        # Usar algoritmo naivo
        NaivStandard(A, B, Result, N, N, N)

def III3SequentialBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método III3SequentialBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    for i1 in range(0, size, bsize):
        for j1 in range(0, size, bsize):
            for k1 in range(0, size, bsize):
                for i in range(i1, min(i1 + bsize, size)):
                    for j in range(j1, min(j1 + bsize, size)):
                        for k in range(k1, min(k1 + bsize, size)):
                            matrizC[i][j] += matrizA[i][k] * matrizB[k][j]

def III4ParallelBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método III4ParallelBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    with ThreadPoolExecutor() as executor:
        for i1 in range(0, size, bsize):
            executor.submit(process_block, matrizA, matrizB, matrizC, size, bsize, i1)

def process_block(matrizA, matrizB, matrizC, size, bsize, i1):
    """
    Procesa un bloque de la matriz.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param i1: El índice de inicio del bloque.
    """
    for j1 in range(0, size, bsize):
        for k1 in range(0, size, bsize):
            for i in range(i1, min(i1 + bsize, size)):
                for j in range(j1, min(j1 + bsize, size)):
                    for k in range(k1, min(k1 + bsize, size)):
                        matrizC[i][j] += matrizA[i][k] * matrizB[k][j]

def IV3SequentialBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método IV3SequentialBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    for i1 in range(0, size, bsize):
        for j1 in range(0, size, bsize):
            for k1 in range(0, size, bsize):
                for i in range(i1, min(i1 + bsize, size)):
                    for j in range(j1, min(j1 + bsize, size)):
                        for k in range(k1, min(k1 + bsize, size)):
                            matrizC[i][k] += matrizA[i][j] * matrizB[j][k]

def IV4ParallelBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método IV4ParallelBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    with ThreadPoolExecutor() as executor:
        for i1 in range(size // bsize):
            executor.submit(calculate_block, matrizA, matrizB, matrizC, size, bsize, i1)

def calculate_block(matrizA, matrizB, matrizC, size, bsize, i1):
    """
    Calcula un bloque de la matriz resultado matrizC.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param i1: Índice para seleccionar el bloque.
    """
    for j1 in range(0, size, bsize):
        for k1 in range(0, size, bsize):
            for i in range(i1 * bsize, min((i1 + 1) * bsize, size)):
                for j in range(j1, min(j1 + bsize, size)):
                    for k in range(k1, min(k1 + bsize, size)):
                        matrizC[i][k] += matrizA[i][j] * matrizB[j][k]

def V3SequentialBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método V3SequentialBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    for i1 in range(0, size, bsize):
        for j1 in range(0, size, bsize):
            for k1 in range(0, size, bsize):
                for i in range(i1, min(i1 + bsize, size)):
                    for j in range(j1, min(j1 + bsize, size)):
                        for k in range(k1, min(k1 + bsize, size)):
                            matrizC[k][i] += matrizA[k][j] * matrizB[j][i]

def V4ParallelBlock(matrizA, matrizB, matrizC, size, bsize, aux):
    """
    Implementación del método V4ParallelBlock para la multiplicación de matrices en Python.

    :param matrizA: La primera matriz.
    :param matrizB: La segunda matriz.
    :param matrizC: La matriz donde se almacenará el resultado de la multiplicación.
    :param size: El tamaño de las matrices.
    :param bsize: El tamaño del bloque.
    :param aux: Un parámetro auxiliar.
    """
    with ThreadPoolExecutor() as executor:
        executor.map(lambda _i: _v4_parallel_block(matrizA, matrizB, matrizC, size, bsize), range(1))

def _v4_parallel_block(matrizA, matrizB, matrizC, size, bsize):
    for i1 in range(0, size, bsize):
        for j1 in range(0, size, bsize):
            for k1 in range(0, size, bsize):
                for i in range(i1, min(i1 + bsize, size)):
                    for j in range(j1, min(j1 + bsize, size)):
                        for k in range(k1, min(k1 + bsize, size)):
                            matrizC[k][i] += matrizA[k][j] * matrizB[j][i]


def III5EnhancedParallelBlock(A, B, size, bsize):
    """
    Multiplica dos matrices A y B usando el algoritmo III_5 Enhanced Parallel Block.

    Args:
        A (list of lists): Matriz A.
        B (list of lists): Matriz B.
        size (int): Tamaño de la matriz.
        bsize (int): Tamaño del bloque.

    Returns:
        list of lists: Matriz resultante de la multiplicación.
    """
    A = np.array(A, dtype=np.int64)
    B = np.array(B, dtype=np.int64)
    C = np.zeros((size, size), dtype=np.int64)

    def process_block(start, end):
        for i1 in range(start, end, bsize):
            for j1 in range(0, size, bsize):
                for k1 in range(0, size, bsize):
                    for i in range(i1, min(i1 + bsize, size)):
                        for j in range(j1, min(j1 + bsize, size)):
                            for k in range(k1, min(k1 + bsize, size)):
                                C[i][j] += A[i][k] * B[k][j]

    with ThreadPoolExecutor() as executor:
        executor.submit(process_block, 0, size // 2)
        executor.submit(process_block, size // 2, size)
    
    return C.tolist()

def matrix_block_multiply(A, B, C, start, end, size, bsize):
    """
    Multiplica bloques de matrices A y B y almacena el resultado en C.

    Args:
        A (numpy.ndarray): Matriz A.
        B (numpy.ndarray): Matriz B.
        C (numpy.ndarray): Matriz de resultado.
        start (int): Índice de inicio.
        end (int): Índice de finalización.
        size (int): Tamaño de la matriz.
        bsize (int): Tamaño del bloque.

    Returns:
        numpy.ndarray: Matriz resultante de la multiplicación.
    """
    for i1 in range(start, end, bsize):
        for j1 in range(0, size, bsize):
            for k1 in range(0, size, bsize):
                for i in range(i1, min(i1 + bsize, size)):
                    for j in range(j1, min(j1 + bsize, size)):
                        for k in range(k1, min(k1 + bsize, size)):
                            C[i, k] += A[i, j] * B[j, k]  # Usando indexación de NumPy
    return C


def IV5EnhancedParallelBlock(A, B, size, bsize):
    """
    Multiplica dos matrices A y B usando el algoritmo IV_5 Enhanced Parallel Block.

    Args:
        A (list of lists): Matriz A.
        B (list of lists): Matriz B.
        size (int): Tamaño de la matriz.
        bsize (int): Tamaño del bloque.

    Returns:
        list of lists: Matriz resultante de la multiplicación.
    """
    A = np.array(A, dtype=np.int64)
    B = np.array(B, dtype=np.int64)
    C = np.zeros((size, size), dtype=np.int64)

    half_size = size // 2

    with ThreadPoolExecutor() as executor:
        # Lanzar dos tareas para multiplicar las partes de la matriz
        future1 = executor.submit(matrix_block_multiply, A, B, C, 0, half_size, size, bsize)
        future2 = executor.submit(matrix_block_multiply, A, B, C, half_size, size, size, bsize)

        # Esperar a que ambas partes completen
        part1 = future1.result()
        part2 = future2.result()

        # No es necesario sumar las partes porque C es compartido
    return C


