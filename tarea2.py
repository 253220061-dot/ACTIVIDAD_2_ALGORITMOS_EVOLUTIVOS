import numpy as np
import pandas as pd
from time import perf_counter

# =====================================================================
# 1. MÓDULO DE DATOS Y CONFIGURACIÓN DEL PROBLEMA TSPTW-P
# =====================================================================
SEED = 42
np.random.seed(SEED)

N_CITIES = 10  # 0: Depósito, 1..9: Clientes
COORDS = np.array([
    [0, 0],      # 0: Depósito
    [10, 20],    # 1
    [30, 40],    # 2
    [20, 10],    # 3
    [50, 10],    # 4
    [60, 50],    # 5
    [40, 60],    # 6
    [15, 70],    # 7
    [70, 20],    # 8
    [25, 30]     # 9
])

# Matriz de distancias euclidianas
DIST_MATRIX = np.zeros((N_CITIES, N_CITIES))
for i in range(N_CITIES):
    for j in range(N_CITIES):
        DIST_MATRIX[i, j] = np.linalg.norm(COORDS[i] - COORDS[j])

TRAVEL_TIME = DIST_MATRIX.copy()
SERVICE_TIME = np.array([0, 5, 5, 5, 5, 5, 5, 5, 5, 5])

# Ventanas de tiempo [e_i, l_i]
TIME_WINDOWS = np.array([
    [0, 300],    # 0 Depósito
    [10, 50],    # 1
    [40, 100],   # 2
    [15, 60],    # 3
    [60, 120],   # 4
    [100, 180],  # 5
    [120, 210],  # 6
    [150, 240],  # 7
    [80, 150],   # 8
    [20, 80]     # 9
])

RHO_PENALTY = 10.0  # Multiplicador de penalización

# =====================================================================
# 2. MÓDULO DE EVALUACIÓN Y DECODIFICACIÓN DE RUTA
# =====================================================================
def evaluar_ruta(route, rho=RHO_PENALTY):
    """
    Decodifica la ruta y calcula las 5 métricas:
    L (distancia), V_TW (violación), T_idle (espera), J (costo penalizado)
    """
    full_route = [0] + list(route) + [0]
    curr_time = 0.0
    total_dist = 0.0
    time_viol = 0.0
    idle_time = 0.0
    arrival_times = [0.0]
    
    for k in range(len(full_route) - 1):
        u = full_route[k]
        v = full_route[k+1]
        
        d = DIST_MATRIX[u, v]
        total_dist += d
        
        arr = curr_time + TRAVEL_TIME[u, v]
        arrival_times.append(arr)
        
        e_v, l_v = TIME_WINDOWS[v]
        
        if arr < e_v:
            idle_time += (e_v - arr)
            curr_time = e_v
        elif arr > l_v:
            time_viol += (arr - l_v)
            curr_time = arr
        else:
            curr_time = arr
            
        curr_time += SERVICE_TIME[v]
        
    total_cost = total_dist + rho * time_viol
    return total_dist, time_viol, idle_time, total_cost, arrival_times

def diversidad_poblacional_rutas(poblacion):
    """Calcula la discrepancia promedio por pares entre permutaciones."""
    N, d = poblacion.shape
    total_diff = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            diff = np.sum(poblacion[i] != poblacion[j]) / d
            total_diff += diff
            count += 1
    return total_diff / count if count > 0 else 0.0

# =====================================================================
# 3. OPERADORES EVOLUTIVOS DE PERMUTACIÓN
# =====================================================================
def inicializar_poblacion_tsptw(n_pop, rng):
    """Muestreo híbrido: RNN + Permutaciones aleatorias."""
    poblacion = []
    # Sembrado heurístico
    for _ in range(5):
        start = rng.integers(1, N_CITIES)
        unvisited = set(range(1, N_CITIES)) - {start}
        r = [start]
        curr = start
        while unvisited:
            nxt = rng.choice(list(unvisited))
            r.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        poblacion.append(np.array(r))
        
    while len(poblacion) < n_pop:
        poblacion.append(rng.permutation(np.arange(1, N_CITIES)))
    return np.array(poblacion)

def seleccionar_ranking_lineal(costos, m, presion_s, rng):
    """Selección por ranking lineal (M11) con asignación de rango medio."""
    n = len(costos)
    orden = np.argsort(costos)
    rangos = np.empty(n)
    rangos[orden] = np.arange(n)
    p = (presion_s - 2.0 * (presion_s - 1.0) * rangos / (n - 1)) / n
    p = np.maximum(p, 0.0)
    p /= p.sum()
    return rng.choice(n, size=m, replace=True, p=p)

def cruce_orden_ox(p1, p2, rng):
    """Order Crossover (OX) para permutaciones."""
    n = len(p1)
    c1, c2 = sorted(rng.choice(n, size=2, replace=False))
    
    h1 = [None] * n
    h1[c1:c2+1] = p1[c1:c2+1]
    p2_fill = [item for item in p2 if item not in h1[c1:c2+1]]
    idx = 0
    for i in range(n):
        if h1[i] is None:
            h1[i] = p2_fill[idx]
            idx += 1
            
    h2 = [None] * n
    h2[c1:c2+1] = p2[c1:c2+1]
    p1_fill = [item for item in p1 if item not in h2[c1:c2+1]]
    idx = 0
    for i in range(n):
        if h2[i] is None:
            h2[i] = p1_fill[idx]
            idx += 1
            
    return np.array(h1), np.array(h2)

def mutacion_inversion_2opt(p, p_m, rng):
    """Inversión de subrutas 2-opt."""
    if rng.random() < p_m:
        n = len(p)
        c1, c2 = sorted(rng.choice(n, size=2, replace=False))
        p_mut = p.copy()
        p_mut[c1:c2+1] = p_mut[c1:c2+1][::-1]
        return p_mut
    return p.copy()

# =====================================================================
# 4. CICLO PRINCIPAL DE EJECUCIÓN DEL ALGORITMO GENÉTICO
# =====================================================================
def ejecutar_ag_tsptw(n_pop=30, t_gen=50, p_c=0.85, p_m=0.30, semilla=42):
    rng = np.random.default_rng(semilla)
    poblacion = inicializar_poblacion_tsptw(n_pop, rng)
    
    evaluaciones = 0
    costos = []
    for ind in poblacion:
        _, _, _, c, _ = evaluar_ruta(ind)
        costos.append(c)
        evaluaciones += 1
    costos = np.array(costos)
    
    mejor_idx = np.argmin(costos)
    mejor_sol = poblacion[mejor_idx].copy()
    mejor_costo = costos[mejor_idx]
    
    historial_costo = [mejor_costo]
    historial_diversidad = [diversidad_poblacional_rutas(poblacion)]
    
    for gen in range(t_gen):
        # Selección
        idx_padres = seleccionar_ranking_lineal(costos, n_pop, 1.7, rng)
        padres = poblacion[idx_padres].copy()
        
        # Variación (Cruce + Mutación)
        hijos = []
        for i in range(0, n_pop, 2):
            p1 = padres[i]
            p2 = padres[(i+1) % n_pop]
            
            if rng.random() < p_c:
                h1, h2 = cruce_orden_ox(p1, p2, rng)
            else:
                h1, h2 = p1.copy(), p2.copy()
                
            h1 = mutacion_inversion_2opt(h1, p_m, rng)
            h2 = mutacion_inversion_2opt(h2, p_m, rng)
            hijos.extend([h1, h2])
            
        hijos = np.array(hijos[:n_pop])
        costos_hijos = []
        for h in hijos:
            _, _, _, c, _ = evaluar_ruta(h)
            costos_hijos.append(c)
            evaluaciones += 1
        costos_hijos = np.array(costos_hijos)
        
        # Reemplazo Generacional con Elitismo (M22, e=2)
        orden_padres = np.argsort(costos)
        elites = poblacion[orden_padres[:2]].copy()
        elites_costos = costos[orden_padres[:2]].copy()
        
        orden_hijos = np.argsort(costos_hijos)
        poblacion = np.vstack((elites, hijos[orden_hijos[:n_pop-2]]))
        costos = np.concatenate((elites_costos, costos_hijos[orden_hijos[:n_pop-2]]))
        
        curr_best = np.argmin(costos)
        if costos[curr_best] < mejor_costo:
            mejor_costo = costos[curr_best]
            mejor_sol = poblacion[curr_best].copy()
            
        historial_costo.append(mejor_costo)
        historial_diversidad.append(diversidad_poblacional_rutas(poblacion))
        
    return mejor_sol, mejor_costo, evaluaciones, historial_costo, historial_diversidad

# Ejecución
mejor_ruta, mejor_costo, evals, h_costo, h_div = ejecutar_ag_tsptw()
l_dist, v_viol, t_idle, j_cost, arr_times = evaluar_ruta(mejor_ruta)

print("=== RESULTADOS DEL ALGORITMO GENÉTICO (TSPTW-P) ===")
print(f"Ruta Óptima Encadenada: 0 -> {' -> '.join(map(str, mejor_ruta))} -> 0")
print(f"Métrica 1 - Distancia Total L(pi): {l_dist:.2f} km")
print(f"Métrica 2 - Violación de Ventana V_TW(pi): {v_viol:.2f} min")
print(f"Métrica 3 - Tiempo de Inactividad T_idle: {t_idle:.2f} min")
print(f"Métrica 4 - Diversidad Final D_route: {h_div[-1]:.4f}")
print(f"Métrica 5 - Costo Penalizado J(pi): {j_cost:.2f}")
print(f"Evaluaciones Totales Consumidas: {evals}")