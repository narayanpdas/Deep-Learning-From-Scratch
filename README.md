# Deep Learning From Scratch

This repository documents my personal journey into the "first principles" of deep learning.

## The Goal

The mission of this repo is to move beyond "black-box" frameworks. I'm building most of the major architectures from the ground up to gain a fundamental understanding of the maths, the logic, and the optimization bottlenecks that frameworks like PyTorch and JAX are built to solve.

This work is split into two pillars:

- **Pillar 1 Fundamental Understanding:** Building models using only Python and NumPy to prove out the core mechanics (like backpropagation).

- **Pillar 2 SOTA Implementation:** Re-implementing state-of-the-art papers (like Transformers) to master advanced architectures and optimization techniques and move to better frameworks once the understanding is clear.

### 1. Fully-Connected Neural Network

- **Status**: _Complete_
- **Stack**: 100% NumPy
- [Go to Project](https://github.com/narayanpdas/Deep-Learning-From-Scratch/tree/main/Neural%20Network%20From%20Scratch)

Summary : A complete neural network built from only NumPy and Maths, demonstrating forward/backward propagation.Contains all the implementation of Loss and Activation Functions.

### 2. Transformer (BERT) (Currently Optimizing!)

- **Status**: _In Progress (Optimization Phase)_
- **Stack**: 100% NumPy (unoptimized, will optimize the Numpy Implementation and then move to Pytorch/Tf)
- [Go to Project](https://github.com/narayanpdas/Deep-Learning-From-Scratch/tree/main/Bert%20From%20Scratch)

Summary : A from-scratch implementation of the BERT-Base architecture. This project breaks down the "Attention Is All You Need" as well as the BERT paper into its functional components, including:

- The full BertModel architecture (based on the official diagram).

- A Tokenizer and DataLoader for processing text.

- All sub-layers: Multi-Head Attention, Feed-Forward Networks, and Positional Embeddings.

- Current Bottleneck: The initial NumPy build is functional but slow (as expected). The next phase is to refactor and optimize this with vectorized operations or using a dedicated framework.
