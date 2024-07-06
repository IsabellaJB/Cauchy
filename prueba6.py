import numpy as np
import math

# ---------------------------------- FUNCION OBJETIVO ---------------------------------- 
def funcion_objetivo(arreglo):
    x = arreglo[0]
    y = arreglo[1]
    operacion = ((x**2 + y - 11)**2) + ((x + y**2 - 7)**2)
    return operacion

# ---------------------------------- GRADIENTE ---------------------------------- 
def gradiente(funcion, x, delta=0.001):
    derivadas = []
    for i in range(len(x)):
        copia = x.copy()
        copia[i] = x[i] + delta
        valor1 = funcion(copia)
        copia[i] = x[i] - delta
        valor2 = funcion(copia)
        derivada = (valor1 - valor2) / (2 * delta)
        derivadas.append(derivada)
    return np.array(derivadas)



# ---------------------------------- BUSQUEDA DORADA ---------------------------------- 
def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2

def w_to_x(w, a, b):
    return w * (b - a) + a

def busquedaDorada(funcion, epsilon, a, b):
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + PHI * Lw
        w1 = bw - PHI * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2



# ---------------------------------- BUSQUEDA DE FIBONACCI ---------------------------------- 
def fibonacci_search(funcion, epsilon, a, b):
    fibs = [1, 1]
    
    while (b - a) / fibs[-1] > epsilon:
        fibs.append(fibs[-1] + fibs[-2])

    n = len(fibs)
    k = n - 2 

    x1 = a + fibs[k-1] / fibs[k+1] * (b - a)
    x2 = a + fibs[k] / fibs[k+1] * (b - a)
    f1 = funcion(x1)
    f2 = funcion(x2)
    
    while k > 1:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + fibs[k] / fibs[k+1] * (b - a)
            f2 = funcion(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + fibs[k-1] / fibs[k+1] * (b - a)
            f1 = funcion(x1)
        k -= 1

    if f1 < f2:
        return x1
    else:
        return x2
    


# ----------------------------------- DISTANCIA ORIGEN ----------------------------------
def distancia_origen(vector):
    return np.linalg.norm(vector)

    


# --------------------------------------- CAUCHY -------------------------------------



def cauchy(funcion_objetivo, x, metodo_busqueda, epsilon1=1e-6, epsilon2=1e-6, max_iterations=100):
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
            alpha = metodo_busqueda(alpha_calcular,epsilon2, 0.0,1.0)
            x_k1 = xk - alpha * gradienteX
            if (distancia_origen(x_k1-xk)/distancia_origen(xk)+0.00001) <= epsilon2:
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk



def redondear(arreglo):
    lita = []
    for valor in arreglo:
        v = round(valor, 2)
        lita.append(v)
    return(lita)


x = [0.0,0.0]





punto_final_golden = (cauchy(funcion_objetivo, x, metodo_busqueda=busquedaDorada))
print(punto_final_golden)
print(f"Resultado con Golden: {redondear(punto_final_golden)}")

print('\n'*2)

punto_final_fibonacci = (cauchy(funcion_objetivo, x, metodo_busqueda=fibonacci_search))
print(punto_final_fibonacci)
print(f"Resultado con Fibonacci: {redondear(punto_final_fibonacci)}")
