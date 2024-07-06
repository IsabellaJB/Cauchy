import numpy as np
import math
from abc import ABC, abstractmethod


class Optimizador(ABC):
    def __init__(self, funcion: callable) -> None:
        self.funcion = funcion

    @abstractmethod
    def optimizar(self):
        pass







class Cauchy(Optimizador):
    def __init__(self, funcion: callable, funcion_objetivo: callable, x: list, epsilon1: float, epsilon2: float, max_iterations: int, alpha: float) -> None:
        super().__init__(funcion)
        self.funcion_objetivo = funcion_objetivo
        self.x = np.array(x)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.max_iterations = max_iterations
        self.alpha = alpha

    def optimizar(self):
        terminar = False
        xk = self.x
        k = 0
        while not terminar:
            gradienteX = np.array(self.gradiente(self.funcion_objetivo, xk))
            distancia = self.distancia_origen(gradienteX)
            if distancia <= self.epsilon1 or k >= self.max_iterations:
                terminar = True
            else:
                alpha_calcular = lambda alpha: self.funcion_objetivo(xk - alpha * gradienteX)
                self.alpha = self.funcion(alpha_calcular, self.epsilon2, 0.0, 1.0)
                x_k1 = xk - self.alpha * gradienteX
                if (self.distancia_origen(x_k1 - xk) / self.distancia_origen(xk) + 0.00001) <= self.epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
        return xk

    @staticmethod
    def gradiente(funcion, x, delta=0.001):
        derivadas = []
        for i in range(len(x)):
            copia = x.copy()
            copia[i] = x[i] + delta
            valor1 = funcion(copia)
            copia[i] = x[i] - delta
            valor2 = funcion(copia)
            valor_final = (valor1 - valor2) / (2 * delta)
            derivadas.append(valor_final)
        return derivadas

    @staticmethod
    def distancia_origen(vector):
        return np.linalg.norm(vector)
























class GoldenSectionSearch(Optimizador):
    def __init__(self, funcion: callable, a: float, b: float, epsilon: float) -> None:
        super().__init__(funcion)
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def optimizar(self):
        PHI = (1 + math.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            aw, bw = self.regla_eliminacion(w1, w2, self.funcion(self.w_to_x(w1, self.a, self.b)),
                                            self.funcion(self.w_to_x(w2, self.a, self.b)), aw, bw)
            Lw = bw - aw
        return (self.w_to_x(aw, self.a, self.b) + self.w_to_x(bw, self.a, self.b)) / 2

    @staticmethod
    def regla_eliminacion(x1, x2, fx1, fx2, a, b):
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2

    @staticmethod
    def w_to_x(w, a, b):
        return w * (b - a) + a






















class Fibonacci(Optimizador):
    def __init__(self, funcion: callable, a: float, b: float, n: int) -> None:
        super().__init__(funcion)
        self.a = a
        self.b = b
        self.n = n

    def optimizar(self):
        a, b = self.a, self.b
        L = b - a
        n = self.n
        k = 2
        bandera = 0

        while bandera != 1:
            i = n - k + 2
            Fi = self.fibonacci(i)

            j = n + 2
            Fj = self.fibonacci(j)

            L_K = (Fi / Fj) * L
            x1 = a + L_K
            x2 = b - L_K

            funcionX1 = self.funcion(x1)
            funcionX2 = self.funcion(x2)

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

        return a, b

    @staticmethod
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b







# Ejemplo de uso:
def funcion_objetivo(arreglo):
    x = arreglo[0]
    y = arreglo[1]
    operacion = ((x ** 2 + y - 11) ** 2) + ((x + y ** 2 - 7) ** 2)
    return operacion


max_iterations = 100
x = [0.0, 0.0]
epsilon1 = 0.001
epsilon2 = 0.001
alpha = 0.2

opt_cauchy = Cauchy(GoldenSectionSearch(funcion_objetivo, 0.0, 1.0, epsilon2).optimizar, funcion_objetivo, x, epsilon1, epsilon2, max_iterations, alpha)
punto_final = opt_cauchy.optimizar()
print(punto_final)

# Redondear el resultado
def redondear(arreglo):
    return [round(valor, 2) for valor in arreglo]

nuevos = redondear(punto_final)
print(nuevos)
