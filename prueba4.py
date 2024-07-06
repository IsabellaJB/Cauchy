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
    def optimizar(self):
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
                alpha = self.metodo_univariable.optimizar(alpha_calcular,0.0,1.0)
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
    def __init__(self, funcion_objetivo, a, b):
        super().__init__(funcion_objetivo)
        self.a = a
        self.b = b
        # self.n = n

    def fibonacci(self, n):
        if n <= 1:
            return n
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def optimizar(self):
        a = self.a
        b = self.b
        L = b - a
        k = 2
        bandera = 0
        while bandera != 1:
            i = self.n - k + 2
            Fi = self.fibonacci(i)
            j = self.n + 2
            Fj = self.fibonacci(j)
            L_K = (Fi / Fj) * L
            x1 = a + L_K
            x2 = b - L_K
            funcionX1 = self.funcion_objetivo(x1)
            funcionX2 = self.funcion_objetivo(x2)
            if funcionX1 > funcionX2:
                a = x1
            elif funcionX1 < funcionX2:
                b = x2
            elif funcionX1 == funcionX2:
                a = x1
                b = x2
            if k == self.n:
                bandera = 1
            else:
                k += 1
        return (a + b) / 2


y = [0.0, 1.0]

opt_golden = GoldenSearch(funcion_objetivo, epsilon=0.001, a=0.0, b=1.0)
opt_cauchy = Cauchy([0, 0], funcion_objetivo, epsilon1=0.001, epsilon2=0.001, M=100, metodo_univariable=opt_golden)
resultado_cauchy = opt_cauchy.optimizar()
print("Resultado Cauchy con Golden Search:", resultado_cauchy)




opt_fibonacci = Fibonacci(funcion_objetivo, a=y[0], b=y[1])
opt_cauchy = Cauchy([0, 0], funcion_objetivo, epsilon1=0.001, epsilon2=0.001, M=100, metodo_univariable=opt_fibonacci)
resultado_cauchy = opt_cauchy.optimizar()
print("Resultado Cauchy con Fibonacci:", resultado_cauchy)



