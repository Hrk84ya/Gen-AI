# Graph Neural Networks (GNNs)

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand graph-structured data and why it matters
- Implement message-passing neural networks from scratch
- Use Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT)
- Apply GNNs to node classification, link prediction, and graph classification
- Know when to use GNNs vs other architectures

## 🕸️ Why Graphs?

Many real-world datasets are naturally graph-structured: social networks, molecules, knowledge graphs, citation networks, road networks, and more. Standard CNNs and MLPs assume grid or sequence structure — GNNs generalize deep learning to arbitrary graph topologies.

### Key Tasks
- **Node Classification**: Predict labels for nodes (e.g., user interests in a social network)
- **Link Prediction**: Predict missing edges (e.g., friend recommendations)
- **Graph Classification**: Classify entire graphs (e.g., molecule toxicity prediction)
- **Graph Generation**: Generate new graphs (e.g., drug discovery)

## 📚 Module Contents

### 1. [GNN Fundamentals](./01_gnn_fundamentals.py)
- Graph representations (adjacency matrix, edge list)
- Message passing framework
- GCN, GraphSAGE, GAT from scratch using PyTorch
- Node classification on Cora citation dataset

### 2. [Advanced GNN Architectures](./02_advanced_gnns.py)
- Graph Isomorphism Network (GIN)
- Graph classification with pooling
- Heterogeneous graphs
- Link prediction

## 📚 Additional Resources

### Papers
- "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
- "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017 — GraphSAGE)
- "Graph Attention Networks" (Veličković et al., 2018)

### Libraries
- [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/)
- [DGL (Deep Graph Library)](https://www.dgl.ai/)

### Online
- [Stanford CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- [Distill.pub — A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/)

---
**Next Module**: [Recommendation Systems](../09-Recommendation-Systems/) →
