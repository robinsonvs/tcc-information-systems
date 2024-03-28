# tcc-information-systems
Neste repositório esta reunido o material referente ao trabalho de conclusão de curso (TCC) na Unisinos, com o tema: Computação Quântica - Análise de algoritmos quânticos na resoluçao do problema SAT na era NISQ.

# Readme: Relações entre o Problema do Clique, Grafos e Redes Sociais

Este README visa estabelecer uma conexão entre o problema do clique, grafos e redes sociais, destacando sua relevância e como eles se relacionam entre si. Além disso, será abordada a relação com a computação quântica e o algoritmo de Grover, especialmente no contexto da resolução de problemas 3-SAT.

## Problema do Clique

O Problema do Clique é um problema fundamental em teoria dos grafos, com diversas aplicações em várias áreas, incluindo redes sociais, biologia computacional, otimização combinatória, entre outras. Em termos simples, o problema consiste em encontrar um subconjunto de vértices em um grafo, onde todos os vértices estão conectados entre si por arestas.

### Conceitos Fundamentais

- **Grafo:** Um grafo é uma estrutura matemática que consiste em um conjunto de vértices (ou nós) conectados por arestas (ou arcos).

- **Clique:** Um clique em um grafo é um subconjunto de vértices onde cada par de vértices está conectado por uma aresta.

### Algoritmo para Encontrar Cliques em um Grafo (em Python)

```python
def find_cliques(graph):
    cliques = []
    for node in graph:
        for clique in cliques:
            if node_neighbours(clique, node, graph):
                clique.append(node)
                break
        else:
            cliques.append([node])
    return [clique for clique in cliques if len(clique) > 1]

def node_neighbours(nodes, node, graph):
    return all(node in graph[neighbour] for neighbour in nodes)
    
#Exemplo de Aplicação

Suponha que temos o seguinte grafo:

Aplicando o algoritmo acima, podemos encontrar todos os cliques neste grafo:    

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}

cliques = find_cliques(graph)
print(cliques)  # Output: [['A', 'B', 'C'], ['B', 'C', 'D']]


Grafos e Redes Sociais

Grafos são estruturas matemáticas que consistem em um conjunto de vértices (ou nós) conectados por arestas (ou arcos). Em muitos contextos, os grafos são usados para modelar e analisar redes sociais. Cada nó pode representar um indivíduo na rede, enquanto as arestas denotam conexões ou relações entre esses indivíduos.
Conceitos Fundamentais

    Redes Sociais: São redes compostas por indivíduos (ou entidades) conectadas por relações sociais, como amizades, seguidores, relações profissionais, etc.

    Nós e Arestas: Em uma rede social, os nós representam os indivíduos e as arestas representam as relações entre eles.

Exemplo de Representação de Rede Social em um Grafo (em Python)

import networkx as nx
import matplotlib.pyplot as plt

# Criando um grafo representando uma rede social

import networkx as nx
import matplotlib.pyplot as plt

# Criando um grafo representando uma rede social
G = nx.Graph()
G.add_edges_from([('Alice', 'Bob'), ('Bob', 'Carol'), ('Carol', 'David'), ('Alice', 'David')])

# Visualizando o grafo
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
plt.show()

Relação com o Problema do Clique

No contexto das redes sociais, o Problema do Clique pode ser interpretado como a busca por grupos de pessoas altamente interconectadas. Encontrar cliques em uma rede social pode ter várias aplicações, como identificar comunidades coesas, grupos de interesses comuns, ou até mesmo identificar potenciais líderes de influência.
Exemplo de Identificação de Cliques em uma Rede Social

Suponha que temos a seguinte rede social:

Podemos usar o algoritmo para encontrar cliques nesta rede social:

social_graph = {
    'Alice': ['Bob', 'Carol', 'David'],
    'Bob': ['Alice', 'Carol'],
    'Carol': ['Alice', 'Bob', 'David'],
    'David': ['Alice', 'Carol']
}

cliques = find_cliques(social_graph)
print(cliques)  # Output: [['Alice', 'Carol', 'David']]


Computação Quântica e Algoritmo de Grover

A computação quântica é um campo emergente que utiliza os princípios da mecânica quântica para processar e armazenar informações de forma diferente da computação clássica. Um dos algoritmos mais conhecidos na computação quântica é o algoritmo de Grover, que oferece uma vantagem significativa na busca em bases de dados não estruturadas.
Conceitos Fundamentais

    Computação Quântica: É um paradigma de computação que utiliza bits quânticos (qubits) para processar informações de maneira diferente da computação clássica.

    Algoritmo de Grover: É um algoritmo quântico que fornece uma vantagem quadrática na busca não estruturada em comparação com algoritmos clássicos.

Exemplo de Implementação do Algoritmo de Grover (em Python)

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram

# Implementação do algoritmo de Grover
def grover_algorithm(oracle, n_qubits):
    grover_circuit = QuantumCircuit(n_qubits)

    # Inicialização dos qubits
    grover_circuit.h(range(n_qubits))

    # Oracle
    grover_circuit.append(oracle, range(n_qubits))

    # Reflexão sobre a média
    grover_circuit.h(range(n_qubits))
    grover_circuit.z(range(n_qubits))
    grover_circuit.cz(0, n_qubits - 1)
    grover_circuit.h(range(n_qubits))

    return grover_circuit

# Simulação do algoritmo de Grover
n = 3  # Número de qubits
oracle = QuantumCircuit(n)
oracle.cz(0, 2)  # Exemplo de oráculo que marca o estado |110>

grover_circuit =


# Comparação de Algoritmos nos Trabalhos Relacionados

|               Paper               |          Model         |          Algorithm         |
|-----------------------------------|------------------------|----------------------------|
| [CHENG; TAO, 2007](./papers/QuantumCooperativeSearchAlgorithmFor3SAT.pdf) | Quantum | Grover |
| [LEPORATI; FELLONI, 2007](./papers/ThreeQuantumAlgorithmsToSolve3SAT.pdf) | Quantum | Grover |
| [CHANG et al., 2008](./papers/QuantumCooperativeSearchAlgorithmFor3SAT.pdf) | Quantum | UREM P Systems |
| [FENG; BLANZIERI; LIANG, 2008](./papers/ImprovedQuantumInspireEvolutionaryAlgorithmAndItsApplicationTo3SATProblems.pdf) | - | Lipton’s DNA-Based |
| [WANG; LIU; LIU, 2020](./papers/AGenericVariableInputsQuantumAlgorithmFor3SATProblem.pdf) | - | QIEA |
| [ALASOW; PERKOWSKI, 2022](./papers/QuantumAlgorithmForMaximumSatisfiability.pdf) | Hybrid | Grover |
| [VARMANTCHAONALA et al., 2023](./papers/QuantumHybridAlgorithmForSolvingSATProblem.pdf) | Hybrid | Grover |


