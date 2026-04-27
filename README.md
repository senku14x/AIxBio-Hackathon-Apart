# Probing Risk Representations in Protein Language Models

**AIxBio Hackathon 2026 | Track 1: DNA Screening & Synthesis Controls (CBAI)**

Submission by : Vishesh Gupta (visheshgupta14x@gmail.com)

This repository contains the experimental codebase for evaluating zero-shot pathogenicity representations in ESM-2 protein language models. It tests the hypothesis that PLM embeddings can serve as supplementary guardrails for DNA synthesis screening, specifically targeting sequences that evade homology-based detection (BLAST).

Read the full paper: https://drive.google.com/file/d/1-q6BbqWRLkQ_mK5X3mG79UB0kL0imVXo/view?usp=sharing

## Biosecurity & Data Availability Statement
Given the dual-use nature of aggregating labelled sequences of Tier 1 and Tier 2 Select Agents (e.g., Ebola, Variola, B. anthracis), **the raw FASTA files and generated embeddings are intentionally excluded from this repository.** 

The `data_pipeline.py` script contains the exact taxonomy queries and UniProt IDs used, allowing researchers to reconstruct the dataset securely via the NCBI Entrez and UniProt APIs.

## Repository Structure

```text
/
├── README.md
└── scripts/
    ├── config.py                 # Global configurations and zero-shot splits
    ├── data_pipeline.py          # Queries NCBI/UniProt to build the dataset
    ├── extract_representations.py# Extracts ESM-2 embeddings (650M / 3B)
    ├── probing.py                # Trains global probes and BLAST baseline
    ├── robustness_suite.py       # Runs UMAP, Fragment, and OOD Artefact tests
    ├── evasion_test.py           # Compares local probes vs BLAST on mutated variants
    ├── evaluate_3b.py            # Executes the 3B parameter scaling ablation
    └── statistical_tests.py      # Computes Bootstrapped CIs and McNemar's p-value

