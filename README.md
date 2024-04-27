# tcc-information-systems
**This repository brings together the material from my course completion work (TCC) at Unisinos - Work Topic: Quantum Computing - Analysis of Quantum Algorithms for Solving the SAT Problem in the NISQ Era**

## Articles used to support the work:

|                          Paper                   |                  Model               |          Algorithm            |
|--------------------------------------------------|--------------------------------------|-------------------------------|
| [CHENG; TAO, 2007](./papers/QuantumCooperativeSearchAlgorithmFor3SAT.pdf)               |      Quantum | Grover         |
| [LEPORATI; FELLONI, 2007](./papers/ThreeQuantumAlgorithmsToSolve3SAT.pdf)               |      Quantum | Grover         |
| [CHANG et al., 2008](./papers/QuantumCooperativeSearchAlgorithmFor3SAT.pdf)             |      Quantum | UREM P Systems |
| [FENG; BLANZIERI; LIANG, 2008](./papers/ImprovedQuantumInspireEvolutionaryAlgorithmAndItsApplicationTo3SATProblems.pdf) | - | Lipton’s DNA-Based |
| [WANG; LIU; LIU, 2020](./papers/AGenericVariableInputsQuantumAlgorithmFor3SATProblem.pdf) | - | QIEA |
| [ALASOW; PERKOWSKI, 2022](./papers/QuantumAlgorithmForMaximumSatisfiability.pdf) | Hybrid | Grover |
| [VARMANTCHAONALA et al., 2023](./papers/QuantumHybridAlgorithmForSolvingSATProblem.pdf) | Hybrid | Grover |


## Introduction

Quantum computing represents a revolutionary approach to computation, leveraging principles from quantum mechanics to process and analyze information in fundamentally new ways. Unlike classical computers, which rely on bits to represent data in binary form (0 or 1), quantum computers use quantum bits or qubits, which can exist in superposition states, enabling them to perform multiple calculations simultaneously.

This experiment aims to explore the capabilities of quantum computing in solving combinatorial optimization problems, specifically focusing on the Maximum Clique Problem (MCP). The MCP is a well-known problem in graph theory, where the objective is to find the largest complete subgraph (clique) within a given graph.

## Background

### Quantum Computing and Classical Computing

Classical computing relies on deterministic algorithms and logical operations to perform computations. It follows the principles of classical physics and operates based on bits that can be in one of two states, 0 or 1. Quantum computing, on the other hand, exploits quantum mechanical phenomena such as superposition and entanglement to perform computations. Qubits can represent both 0 and 1 simultaneously, allowing quantum computers to process vast amounts of data in parallel.

### Quantum Mechanics and Computation

Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the smallest scales, such as atoms and subatomic particles. It provides the foundation for quantum computing, as qubits operate according to quantum principles such as superposition, entanglement, and interference. These phenomena enable quantum computers to solve certain problems much more efficiently than classical computers.

## Objectives of the Experiment

The main objective of this experiment is to showcase the potential of quantum computing in solving complex combinatorial optimization problems, particularly focusing on leveraging the Grover algorithm to tackle the Boolean satisfiability (SAT) problem. By utilizing quantum principles, we aim to demonstrate how quantum computers can efficiently search for valid solutions in large solution spaces.

### Specific Goals:

1. **Application of the Grover Algorithm:** The primary focus of this experiment is to apply the Grover algorithm, a powerful quantum algorithm for searching unstructured databases, to efficiently search for solutions to the SAT problem. By harnessing the quantum parallelism and amplitude amplification properties of the Grover algorithm, we aim to demonstrate its superiority over classical search algorithms in finding solutions to SAT instances.

2. **Reduction of the Maximum Clique to SAT:** The graph-based Maximum Clique Problem (MCP) will serve as a real-world problem scenario, where the goal is to identify the largest subset of individuals in a social network who are all directly connected to each other. By reducing the MCP to a Boolean satisfiability problem (SAT), we will create SAT instances that represent the maximum clique problem, allowing us to apply the Grover algorithm to search for valid solutions.

3. **Validation of Quantum Solution:** Through this experiment, we seek to validate the quantum solution obtained from the Grover algorithm by comparing it with classical brute-force search methods. By demonstrating the efficiency and effectiveness of quantum algorithms in solving SAT instances derived from real-world problems, we aim to showcase the potential impact of quantum computing in addressing complex optimization challenges.


## Methodology

### 1. Generating a Complete Graph

A complete graph is a type of graph in which each pair of vertices is connected by a unique edge. In this experiment, we will generate a complete graph where the vertices represent elements, and the edges represent relationships between these elements. The complete graph can be generated randomly or based on specific criteria, depending on the problem context.

### 2. Identifying the Maximum Clique

The maximum clique in a graph is a subset of vertices in which every pair of vertices is connected by an edge. The identification of the maximum clique involves applying specific graph search algorithms, such as the Bron-Kerbosch algorithm or algorithms based on integer programming, to find the set of vertices that forms the largest clique in the graph.

### 3. Reduction of the Clique to SAT

After identifying the maximum clique, the next step is to reduce it to a formula in Conjunctive Normal Form (CNF), which is a standard representation for Boolean satisfiability (SAT) problems. This reduction ensures that the solution to the maximum clique problem is equivalent to the solution to the Boolean satisfiability problem.

### 4. Application of the Grover Algorithm

The Grover algorithm is a quantum algorithm used for searching an unstructured list of items. It is known for its ability to search for solutions quadratically, making it efficient for certain search problems, including the Boolean satisfiability problem (SAT). We will use the Grover algorithm to search for valid solutions to the CNF formula generated in the previous step, thereby validating the solutions of the maximum clique.

## Conclusion

By following this methodology, we aim to demonstrate the feasibility and effectiveness of quantum algorithms, such as the Grover algorithm, in solving combinatorial optimization problems. Each step of the process is essential for the success of the experiment, from generating the complete graph to applying the Grover algorithm to find valid solutions to the maximum clique problem.


* ## [Clique Problem](https://cs.stanford.edu/people/eroberts/courses/soco/projects/2003-04/dna-computing/clique.htm)
* ## [Grover Algorithm](https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/grovers-algorithm)



