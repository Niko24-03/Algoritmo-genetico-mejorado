
import subprocess
import sys
import os
import random
import copy
import time
import blosum
import numpy as np
import matplotlib.pyplot as plt
import re

# ------------------------
# CONFIGURACIÓN PRINCIPAL
# ------------------------
NP_OBL_DEFAULT = 50         # tamaño de población por defecto (mayor diversidad)
GENERACIONES_DEFAULT = 80   # generaciones por defecto
N_CORTES_DEFAULT = 3        # cruza multipunto
ELITISMO_DEFAULT = 3
TORNEO_K_DEFAULT = 3
REFRESCO_LIMITE_DEFAULT = 10

blosum62 = blosum.BLOSUM(62)
NFE = 0

# ------------------------
# DATOS ORIGINALES
# ------------------------
def get_sequences():
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

# ------------------------
# FUNCIONES AUXILIARES
# ------------------------

def crear_poblacion_inicial(n=20):
    base = get_sequences()
    return [[row[:] for row in base] for _ in range(n)]

def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(row) for row in individuo)
    return [row + [gap] * (max_len - len(row)) for row in individuo]

# ------------------------
# EVALUACIÓN OPTIMIZADA
# ------------------------

def evaluar_individuo_blosum62(individuo):
    """Evalúa un individuo usando BLOSUM62. Versión optimizada con NumPy para acelerar.

    - individuo: lista de listas (secuencias con '-')
    - devuelve: score (float)
    """
    global NFE
    NFE += 1
    n = len(individuo)
    L = len(individuo[0])
    arr = np.array(individuo)  # shape (n, L)

    # Penalización por gaps: para cada columna, cada par que tenga al menos un gap penaliza -4
    # Calculamos para cada par de secuencias cuántas columnas tienen gaps en cualquiera de ellas
    # Usamos máscara por pares
    score = 0.0

    # Par vectorizado: iteramos por pares (n pequeño: 3 secuencias típicamente)
    for i in range(n):
        for j in range(i + 1, n):
            ai = arr[i]
            aj = arr[j]
            mask = (ai != '-') & (aj != '-') # columnas sin gaps en ambos
            if np.any(mask):
                # convertimos a listas y sumamos (acceso por matriz blosum)
                score += sum(blosum62[a][b] for a, b in zip(ai[mask], aj[mask]))
            # penalizaciones por columnas donde al menos uno es gap
            ngaps = np.sum(~mask)
            score -= 4 * ngaps

    return score

# ------------------------
# OPERADORES GENÉTICOS
# ------------------------

def mutar_individuo_avanzado(individuo, p_insert=0.25, p_mover=0.2, max_bloque=3):
    nuevo = []
    for seq in individuo:
        s = seq[:]
        if random.random() < p_insert:
            pos = random.randint(0, len(s))
            bloque = ['-'] * random.randint(1, max_bloque)
            s[pos:pos] = bloque
        if random.random() < p_mover and '-' in s:
            idxs = [i for i, x in enumerate(s) if x == '-']
            i = random.choice(idxs)
            s.pop(i)
            pos = random.randint(0, len(s))
            s.insert(pos, '-')
        nuevo.append(s)
    return nuevo


def cruzar_individuos_multipunto_seguro(ind1, ind2, n_cortes=3):
    hijo1, hijo2 = [], []
    for seq1, seq2 in zip(ind1, ind2):
        aa1 = [a for a in seq1 if a != '-']
        aa2 = [a for a in seq2 if a != '-']
        if len(aa1) <= n_cortes or len(aa2) <= n_cortes:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue

        cortes = sorted(random.sample(range(1, min(len(aa1), len(aa2)) - 1), n_cortes))
        nueva1, nueva2 = aa1[:], aa2[:]
        for i in range(0, len(cortes) - 1, 2):
            p1, p2 = cortes[i], cortes[i + 1]
            nueva1[p1:p2], nueva2[p1:p2] = nueva2[p1:p2], nueva1[p1:p2]

        def reinsertar_gaps(original, sin_gaps):
            res = []
            j = 0
            for a in original:
                if a == '-':
                    res.append('-')
                else:
                    res.append(sin_gaps[j])
                    j += 1
            return res

        hijo1.append(reinsertar_gaps(seq1, nueva1))
        hijo2.append(reinsertar_gaps(seq2, nueva2))

    hijo1 = mutar_individuo_avanzado(hijo1, p_insert=0.2, p_mover=0.15)
    hijo2 = mutar_individuo_avanzado(hijo2, p_insert=0.2, p_mover=0.15)
    return hijo1, hijo2


def cruzar_poblacion_segura(poblacion, n_cortes=3):
    nueva = []
    n = len(poblacion)
    idxs = list(range(n))
    random.shuffle(idxs)
    parejas = [(idxs[i], idxs[i + 1]) for i in range(0, n - 1, 2)]
    if n % 2 == 1:
        parejas.append((idxs[-1], idxs[0]))
    for i1, i2 in parejas:
        p1, p2 = poblacion[i1], poblacion[i2]
        h1, h2 = cruzar_individuos_multipunto_seguro(p1, p2, n_cortes=n_cortes)
        nueva += [copy.deepcopy(p1), copy.deepcopy(p2), h1, h2]
    return nueva[:2 * n]

# ------------------------
# SELECCIÓN Y ELITISMO
# ------------------------

def seleccion_torneo(poblacion, scores, k=3):
    seleccionados = []
    for _ in range(len(poblacion)):
        competidores = random.sample(list(zip(poblacion, scores)), k)
        ganador = max(competidores, key=lambda x: x[1])[0]
        seleccionados.append(copy.deepcopy(ganador))
    return seleccionados


def aplicar_elitismo(poblacion, scores, elite_size=2):
    mejores_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_size]
    return [copy.deepcopy(poblacion[i]) for i in mejores_idx]

# ------------------------
# REFRESCO Y VALIDACIÓN
# ------------------------

def refrescar_poblacion(poblacion, generaciones_sin_mejora, limite=10, fraccion_nuevos=0.25):
    if generaciones_sin_mejora >= limite:
        n_nuevos = max(1, int(len(poblacion) * fraccion_nuevos))
        nuevos = crear_poblacion_inicial(n_nuevos)
        print(f"[Refresco] {n_nuevos} nuevos individuos tras {generaciones_sin_mejora} generaciones sin mejora.")
        poblacion.extend(nuevos)
    return poblacion


def validar_poblacion_sin_gaps(poblacion, originales):
    for ind in poblacion:
        for seq, orig in zip(ind, originales):
            sin_gaps = [a for a in seq if a != '-']
            if sin_gaps != orig:
                return False
    return True

# Versión mejorada del AG
def run_improved(n_poblacion=NP_OBL_DEFAULT, generaciones=GENERACIONES_DEFAULT, n_cortes=N_CORTES_DEFAULT,
                 elitismo_size=ELITISMO_DEFAULT, torneo_k=TORNEO_K_DEFAULT, refresco_limite=REFRESCO_LIMITE_DEFAULT,
                 verbose=False):
    global NFE
    NFE = 0
    originales = get_sequences()
    poblacion = crear_poblacion_inicial(n_poblacion)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]

    best, fitness_best = obtener_best(scores, poblacion)
    veryBest, fitnessVeryBest = best, fitness_best
    generaciones_sin_mejora = 0

    historial = []
    start = time.time()

    for gen in range(generaciones):
        elite = aplicar_elitismo(poblacion, scores, elite_size=elitismo_size)
        poblacion = cruzar_poblacion_segura(poblacion, n_cortes=n_cortes)
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion = seleccion_torneo(poblacion, scores, k=torneo_k)
        poblacion.extend(elite)

        # controlar tamaño fijo
        if len(poblacion) > n_poblacion:
            poblacion = poblacion[:n_poblacion]

        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]

        best, fitness_best = obtener_best(scores, poblacion)
        if fitness_best > fitnessVeryBest:
            fitnessVeryBest = fitness_best
            veryBest = copy.deepcopy(best)
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        poblacion = refrescar_poblacion(poblacion, generaciones_sin_mejora, limite=refresco_limite)

        if not validar_poblacion_sin_gaps(poblacion, originales):
            print("[ALERTA] Validación falló durante ejecución mejorada. Se interrumpe para depuración.")
            break

        historial.append(fitness_best)
        if verbose and gen % 5 == 0:
            print(f"Gen {gen:03d} | Best: {fitness_best:.2f} | VeryBest: {fitnessVeryBest:.2f} | NFE: {NFE}")

    tiempo = time.time() - start
    return {
        'historial': historial,
        'veryBest': veryBest,
        'fitnessVeryBest': fitnessVeryBest,
        'NFE': NFE,
        'tiempo': tiempo
    }

# ------------------------
# EJECUCIÓN: VERSIÓN ORIGINAL (AG10.py) vía subprocess
# ------------------------

def run_original_via_subprocess(path_ag10='AG10.py', timeout=300):
    """Ejecuta AG10.py como proceso separado y captura el output.

    Devuelve diccionario con historial (lista de best por gen) y tiempo total.
    """
    if not os.path.exists(path_ag10):
        raise FileNotFoundError(f"No se encuentra {path_ag10} en el directorio actual: {os.getcwd()}")

    start = time.time()
    proc = subprocess.Popen([sys.executable, path_ag10], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    historial = []
    best_final = None
    try:
        pattern = re.compile(r"fitness: \s*([-0-9]+\.?[0-9]*)")
        for line in proc.stdout:
            line = line.strip()
            print("[AG10] ", line)
            m = pattern.search(line)
            if m:
                try:
                    val = float(m.group(1))
                    historial.append(val)
                    best_final = val if best_final is None or val > best_final else best_final
                except:
                    pass
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[AG10] Proceso original excedió timeout y fue terminado.")
    tiempo = time.time() - start
    return {
        'historial': historial,
        'fitnessFinal': best_final,
        'tiempo': tiempo
    }

# ------------------------
# UTILIDADES
# ------------------------

def obtener_best(scores, poblacion):
    idx = scores.index(max(scores))
    return copy.deepcopy(poblacion[idx]), scores[idx]

# ------------------------
# PLOTEO
# ------------------------

def plot_results(hist_improved, hist_original, tiempo_improved, tiempo_original):
    # Asegurar mismas longitudes para la comparacion de series (rellenar con ultimo valor)
    L = max(len(hist_improved), len(hist_original))
    himp = hist_improved + [hist_improved[-1]] * (L - len(hist_improved)) if hist_improved else [0]*L
    horig = hist_original + [hist_original[-1]] * (L - len(hist_original)) if hist_original else [0]*L

    gens = list(range(L))

    # Figura 1: Evolución del fitness por generación
    plt.figure(figsize=(10, 5))
    plt.plot(gens, horig, label='Original (AG10)', linestyle='--')
    plt.plot(gens, himp, label='Mejorada (AG11d)', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Fitness (mejor de la generación)')
    plt.title('Evolución del fitness: AG10 vs AG11d')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparacion_fitness.png')
    plt.show()

    # Figura 2: Comparación de tiempos (barra)
    plt.figure(figsize=(6,4))
    plt.bar(['AG10 (original)', 'AG11d (mejorada)'], [tiempo_original, tiempo_improved], color=['orange','steelblue'])
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de tiempo total')
    plt.tight_layout()
    plt.savefig('comparacion_tiempos.png')
    plt.show()
#Configuración final y ejecución
#comparación entre AG10 y AG11_mejorado

def main(run_AG10=True, n_poblacion=NP_OBL_DEFAULT, generaciones=GENERACIONES_DEFAULT):
    print("AG11d: ejecución de la versión mejorada (NumPy, población fija)")
    improved = run_improved(n_poblacion=n_poblacion, generaciones=generaciones, n_cortes=N_CORTES_DEFAULT,
                             elitismo_size=ELITISMO_DEFAULT, torneo_k=TORNEO_K_DEFAULT, refresco_limite=REFRESCO_LIMITE_DEFAULT,
                             verbose=True)
    hist_imp = improved['historial']
    t_imp = improved['tiempo']

    hist_orig = []
    t_orig = 0.0
    if run_AG10:
        print("\nEjecutando AG10.py (original) para comparar — esto puede tardar dependiendo de su implementación...")
        try:
            orig = run_original_via_subprocess('AG10.py', timeout=600)
            hist_orig = orig['historial']
            t_orig = orig['tiempo']
        except Exception as e:
            print("Error al ejecutar AG10.py:", e)

    # Si la versión original no generó historial, creamos uno dummy para evitar fallos
    if not hist_orig:
        hist_orig = [hist_imp[0]] * len(hist_imp) if hist_imp else []

    # Graficar resultados
    plot_results(hist_imp, hist_orig, t_imp, t_orig)
    print('Gráficas guardadas: comparacion_fitness.png, comparacion_tiempos.png')

if __name__ == '__main__':
    # Ejecutar con población mayor según tu solicitud
    main(run_AG10=True, n_poblacion=50, generaciones=80)