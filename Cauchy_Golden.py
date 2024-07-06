
import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC



def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2


def w_to_x(w, a, b):
    return w*(b-a) + a


def busquedaDorada(funcion, epsilon, a, b):
    PHI = (1+math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw - PHI*Lw
        aw, bw = regla_eliminacion(w1,w2, funcion(w_to_x(w1,a,b)),
                                   funcion(w_to_x(w2,a,b)), aw, bw)
        k+=1
        Lw = bw - aw
    return (w_to_x(aw, a, b)+w_to_x(bw,a,b))/2








def fibonacci(n):
    if n <= 1:
        return n
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

def fibonacciSearch(x, funcion):
    a = x[0]
    b = x[-1]
    L = b - a
    n = 7
    k = 2
    bandera = 0
    # paso 2
    while (bandera != 1):
        i = n - k + 2
        Fi = fibonacci(i)
        j = n + 2
        Fj = fibonacci(j)
        L_K = (Fi/Fj) * L
        x1 = a + L_K
        x2 = b - L_K
        funcionX1 = funcion(x1)
        funcionX2 = funcion(x2)
        if funcionX1 > funcionX2:
            a = x1
        elif funcionX1 < funcionX2:
            b = x2
        elif funcionX1 == funcionX2:
            a = x1
            b = x2
        if k == n:
            bandera = 1
        else:
            k += 1
    return ((a+b)/2)














def funcion_objetivo(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

def gradiente(funcion, x, delta=0.001):
    derivadas = []
    for i in range (0, len(x)):
        valor1 = 0
        valor2 = 0
        valor_final = 0
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        copia[i] = x[i] - delta
        valor2 = funcion(copia)
        valor_final = (valor1 - valor2) / (2*delta)
        derivadas.append(valor_final)
    return derivadas

def distancia_origen(vector):
    return np.linalg.norm(vector)

def redondear(arreglo):
    lita = []
    for valor in arreglo:
        v = round(valor, 2)
        lita.append(v)
    return(lita)

def cauchy(funcion, funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha):
    terminar = False
    xk = x
    k = 0
    while not terminar:
        gradienteX = np.array(gradiente(funcion_objetivo,xk))
        distancia = distancia_origen(gradienteX)
        if distancia <= epsilon1:
            terminar = True
        elif (k >= max_iterations):
            terminar = True
        else:
            def alpha_calcular(alpha):
                return funcion_objetivo(xk - alpha*gradienteX)
            alpha = funcion(alpha_calcular,epsilon2, 0.0,1.0)
            x_k1 = xk - alpha * gradienteX
            if (distancia_origen(x_k1-xk)/distancia_origen(xk)+0.00001) <= epsilon2:
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk



max_iterations = 100
x = [0.0,0.0]
deltaX = 0.01

epsilon1 = 0.001
epsilon2 = 0.001
k = 0
alpha = 0.2





punto_final = (cauchy(busquedaDorada, funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha))
print(punto_final)

nuevos = redondear(punto_final)
print(nuevos)

