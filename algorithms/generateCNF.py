def is_edge(graph, u, v):
    return u in graph[v] or v in graph[u]

def generate_clauses(graph, k):
    n = len(graph)
    clauses = []

    # Cláusulas para garantir que k vértices são selecionados
    for i in range(1, k + 1):
        clauses.append([j + n * (i - 1) for j in range(1, n + 1)])

    # Cláusulas para garantir que não há dois vértices no mesmo slot
    for i in range(1, k + 1):
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                clauses.append([-(u + n * (i - 1)), -(v + n * (i - 1))])

    # Cláusulas para garantir que todos os pares de vértices no clique são adjacentes
    for i in range(1, k):
        for j in range(i + 1, k + 1):
            for u in range(1, n + 1):
                for v in range(1, n + 1):
                    if not is_edge(graph, u - 1, v - 1):
                        clauses.append([-(u + n * (i - 1)), -(v + n * (j - 1))])

    return clauses



def main():
    # 5 vertices k=5
    graph = {
        0: [1, 2, 3, 4],
        1: [0, 2, 3, 4],
        2: [0, 1, 3, 4],
        3: [0, 1, 2, 4],
        4: [0, 1, 2, 3],
    }    
        
    k = 5
    n = 5
    
    clauses = generate_clauses(graph, k)
    
    print("Generated clauses:")
    for clause in clauses:
        print(clause)

if __name__ == "__main__":
    main()
