

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime

def is_edge(graph, u, v):
    return u in graph[v] or v in graph[u]

def generate_clauses(graph, k):
    n = len(graph)
    clauses = []
    
    # Cláusulas para garantir que k vértices são selecionados
    for i in range(1, k+1):
        clauses.append([j + n * (i - 1) for j in range(1, n+1)])
    
    # Cláusulas para garantir que não há dois vértices no mesmo slot
    for i in range(1, k+1):
        for u in range(1, n+1):
            for v in range(u + 1, n+1):
                clauses.append([-(u + n * (i - 1)), -(v + n * (i - 1))])
    
    # Cláusulas para garantir que todos os pares de vértices no clique são adjacentes
    for i in range(1, k):
        for j in range(i+1, k+1):
            for u in range(1, n+1):
                for v in range(1, n+1):
                    if not is_edge(graph, u-1, v-1):
                        clauses.append([-(u + n * (i - 1)), -(v + n * (j - 1))])

    return clauses



def calculate_optimal_iterations(n, M=1):
    N = 2**n
    return int(np.round(np.pi / 4 * np.sqrt(N / M)))



if __name__ == '__main__':

    # 3 vertices k=3
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }      


    # # 4 vertices k=4
    # graph = {
    #     0: [1, 2, 3],
    #     1: [0, 2, 3],
    #     2: [0, 1, 3],
    #     3: [0, 1, 2],
    # }       

    # 5 vertices k=5
    # graph = {
    #     0: [1, 2, 3, 4],
    #     1: [0, 2, 3, 4],
    #     2: [0, 1, 3, 4],
    #     3: [0, 1, 2, 4],
    #     4: [0, 1, 2, 3],
    # }    

    

    K = 3
    n = len(graph) * K

    cnf_clauses = generate_clauses(graph, K)
    print(cnf_clauses)
    print("***************")

    expression = ''
    for i, clause in enumerate(cnf_clauses):
        if i > 0:
            expression += ' & '
        expression += '(' + ' | '.join(['x' + str(x) if x > 0 else '~x' + str(abs(x)) for x in clause]) + ')'
    print(expression)
    print("***************")

    # Apply "optimized" Grover 
    from qiskit.circuit.library import PhaseOracle
    from qiskit.primitives import Sampler
    from qiskit.visualization import plot_histogram
    from qiskit_algorithms import AmplificationProblem, Grover

    start_time = datetime.now()

    print("starting ....... ")

    iteracoes = int((math.pi / 4) * math.sqrt((2 ** n) / 1))

    print("phaseoracle...")
    oracle = PhaseOracle(expression)
    print("amplification...")
    problem = AmplificationProblem(oracle)
    print("grover...")
    grover = Grover(sampler=Sampler(), iterations=iteracoes)
    print("amplify...")
    results = grover.amplify(problem)

    end_time = datetime.now()

    print("ending ......... ")

    elapsed_time = end_time - start_time

    #print("oracle")
    #cplot1=grover.draw("mpl", style = "iqp")
    #display(cplot1)

    print('É Satisfatível?', results.oracle_evaluation)
    print('Estado amplificado', results.top_measurement)
    print('Probabilidade estado amplificado', results.max_probability)
    print('Iterações máximas', iteracoes)
    print('Width', problem.grover_operator.decompose().width())
    print('Depth', problem.grover_operator.decompose().depth())
    print('Data/hora de início:', start_time)
    print('Data/hora de fim:', end_time)
    print('Tempo decorrido:', elapsed_time)    

    # Gerar somente resultados relevantes
    output = dict()
    for x in results.circuit_results[0]:
        if x:
            value = results.circuit_results[0][x]
            if value >= results.max_probability - results.max_probability * 0.005:
                output.update({ x:  value })
                print(f"estado_amplificado: {x} probabilidade {value}")

    # Ordenar o output por probabilidade
    sorted_output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))

    # Plotar o histograma
    plt.bar(sorted_output.keys(), sorted_output.values(), width=0.5)
    plt.xticks(fontsize=8, rotation=45)
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.title('Histogram of Amplified States')
    plt.ylim(0, max(sorted_output.values()) * 1.1)  # Ajustar o limite superior do eixo y

    # Adicionar as probabilidades sobre as barras
    for i, (state, probability) in enumerate(sorted_output.items()):
        plt.text(i, probability, f'{probability:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)

    plt.show()
    #input()
