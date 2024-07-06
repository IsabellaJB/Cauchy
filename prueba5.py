class Optimizador:
    def __init__(self, funcion_objetivo):
        self.funcion_objetivo = funcion_objetivo

    def optimizar(self):
        raise NotImplementedError("Este método debe ser sobrescrito por las subclases")


class Cauchy(Optimizador):
    def __init__(self, funcion_objetivo, epsilon1, epsilon2, M, metodo_univariable):
        super().__init__(funcion_objetivo)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.M = M
        self.metodo_univariable = metodo_univariable

    def optimizar(self):
        # Aquí iría la implementación del método Cauchy
        resultado = self.metodo_univariable.optimizar()
        return resultado


class GoldenSearch(Optimizador):
    def __init__(self, funcion_objetivo, epsilon, a, b):
        super().__init__(funcion_objetivo)
        self.epsilon = epsilon
        self.a = a
        self.b = b

    def optimizar(self):
        # Implementación del método Golden Search
        phi = (1 + 5 ** 0.5) / 2  # Constante de oro
        resphi = 2 - phi

        a, b = self.a, self.b
        c = b - resphi * (b - a)
        d = a + resphi * (b - a)
        
        while abs(c - d) > self.epsilon:
            if self.funcion_objetivo(c) < self.funcion_objetivo(d):
                b = d
            else:
                a = c

            c = b - resphi * (b - a)
            d = a + resphi * (b - a)

        return (b + a) / 2


class Fibonacci(Optimizador):
    def __init__(self, funcion_objetivo, a, b, n):
        super().__init__(funcion_objetivo)
        self.a = a
        self.b = b
        self.n = n

    def optimizar(self):
        # Implementación del método Fibonacci
        fib = [0, 1]
        for i in range(2, self.n+1):
            fib.append(fib[-1] + fib[-2])
        
        a, b = self.a, self.b
        k = 0
        while k < (self.n - 2):
            Lk = (fib[self.n - k - 1] / fib[self.n - k]) * (b - a)
            c = a + Lk
            d = b - Lk
            if self.funcion_objetivo(c) < self.funcion_objetivo(d):
                b = d
            else:
                a = c
            k += 1
        
        return (b + a) / 2


# Ejemplo de uso:
def ejemplo_funcion_objetivo(x):
    return (x - 2)**2  # Solo un ejemplo de función

# Crear instancias de cada optimizador con parámetros de ejemplo
opt_golden = GoldenSearch(ejemplo_funcion_objetivo, epsilon=0.01, a=0, b=4)
opt_fibonacci = Fibonacci(ejemplo_funcion_objetivo, a=0, b=4, n=10)

# Crear una instancia del método Cauchy con GoldenSearch como subcomponente
opt_cauchy = Cauchy(ejemplo_funcion_objetivo, epsilon1=0.01, epsilon2=0.01, M=100, metodo_univariable=opt_golden)

# Llamar al método optimizar
resultado_cauchy = opt_cauchy.optimizar()
print("Resultado Cauchy con Golden Search:", resultado_cauchy)
