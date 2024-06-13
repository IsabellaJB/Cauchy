import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def funcion_objetivo(arreglo):
    x = arreglo[0]
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

class Optimizador(ABC):
    def __init__(self, funcion_objetivo):
        self.funcion_objetivo = funcion_objetivo

    @abstractmethod
    def optimizar(self, *args):
        pass

class Cauchy(Optimizador):
    def __init__(self, x, funcion_objetivo, epsilon1, epsilon2, M, metodo_univariable):
        super().__init__(funcion_objetivo)
        self.x = x
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.M = M
        self.metodo_univariable = metodo_univariable

    def gradiente(self, x, delta=0.001):
        derivadas = []
        for i in range(len(x)):
            copia = x.copy()
            copia[i] = x[i] + delta
            valor1 = self.funcion_objetivo(copia)
            copia[i] = x[i] - delta
            valor2 = self.funcion_objetivo(copia)
            valor_final = (valor1 - valor2) / (2 * delta)
            derivadas.append(valor_final)
        return derivadas

    def distancia_origen(self, vector):
        return np.linalg.norm(vector)

    def redondear(self, arreglo):
        return [round(valor, 2) for valor in arreglo]

    def optimizar(self):
        terminar = False
        xk = self.x
        k = 0
        while not terminar:
            gradienteX = np.array(self.gradiente(xk))
            distancia = self.distancia_origen(gradienteX)
            if distancia <= self.epsilon1:
                terminar = True
            elif k >= self.M:
                terminar = True
            else:
                def alpha_calcular(alpha):
                    return self.funcion_objetivo(xk - alpha * gradienteX)
                alpha = self.metodo_univariable.optimizar(alpha_calcular, 0.0, 1.0)
                x_k1 = xk - alpha * gradienteX
                if (self.distancia_origen(x_k1 - xk) / (self.distancia_origen(xk) + 0.00001)) <= self.epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
        return xk

class GoldenSearch(Optimizador):
    def __init__(self, funcion_objetivo, epsilon, a, b):
        super().__init__(funcion_objetivo)
        self.epsilon = epsilon
        self.a = a
        self.b = b

    def regla_eliminacion(self, x1, x2, fx1, fx2, a, b):
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2

    def w_to_x(self, w, a, b):
        return w * (b - a) + a

    def optimizar(self, funcion, a, b):
        PHI = (1 + math.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            aw, bw = self.regla_eliminacion(
                w1, w2,
                funcion(self.w_to_x(w1, a, b)),
                funcion(self.w_to_x(w2, a, b)),
                aw, bw
            )
            Lw = bw - aw
        return (self.w_to_x(aw, a, b) + self.w_to_x(bw, a, b)) / 2

class Fibonacci(Optimizador):
    def __init__(self, funcion_objetivo, epsilon):
        super().__init__(funcion_objetivo)
        self.epsilon = epsilon

    def fibonacci(self, n):
        if n <= 1:
            return n
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def optimizar(self, funcion, a, b):
        L = b - a
        n = 0
        while self.fibonacci(n) < (b - a) / self.epsilon:
            n += 1
        k = 0
        while k < (n - 2):
            L_k = self.fibonacci(n - k - 1) / self.fibonacci(n - k + 1) * L
            x1 = a + L_k
            x2 = b - L_k
            if funcion(x1) > funcion(x2):
                a = x1
            else:
                b = x2
            k += 1
        return (a + b) / 2

def ejemplo_funcion_objetivo(x):
    return (x - 2)**2

opt_golden = GoldenSearch(ejemplo_funcion_objetivo, epsilon=0.01, a=0, b=4)
opt_fibonacci = Fibonacci(ejemplo_funcion_objetivo, epsilon=0.01)
opt_cauchy = Cauchy([0, 0], funcion_objetivo, epsilon1=0.01, epsilon2=0.01, M=100, metodo_univariable=opt_fibonacci)

resultado_cauchy = opt_cauchy.optimizar()
print("Resultado Cauchy con Fibonacci:", resultado_cauchy)
