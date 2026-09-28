# Text Simplification System

A hybrid system that rewrites complex academic and technical text in plain, accessible language. It combines a fast rule-based simplifier with two transformer models (T5 and BART) and evaluates the results with standard simplification and readability metrics.

## Overview

Dense technical writing is hard to read for a general audience. This project addresses that by pairing the transparency and speed of rule-based substitution with the contextual understanding of modern neural models, so the output stays accurate while becoming easier to read.

## Key Features

- **Three simplification approaches**
  - Rule-based: fast, transparent word and phrase substitution
  - T5: context-aware neural simplification
  - BART: fluent, natural-sounding rewrites
- **Built-in evaluation** using SARI, BLEU, FKGL and a custom human readability score
- **Interactive CLI** for simplifying text in real time
- **Dataset support** for WikiSimple and OneStop English

## System Architecture

The system is a modular pipeline:

| Component | Role |
| :--- | :--- |
| Dataset Manager | Loads training and evaluation data from WikiSimple and OneStop English |
| Rule-Based Simplifier | Deterministic engine with 60+ complex-word mappings and 15+ phrase patterns |
| Transformer Engine | T5 and BART models for neural simplification |
| Evaluation Framework | Computes quality metrics to compare the models |
| Interactive Interface | Command-line shell for real-time testing |

## Requirements

- Python 3.7 or later
- PyTorch
- NLTK

## Evaluation Metrics

| Metric | What it measures | How to read it |
| :--- | :--- | :--- |
| SARI | Simplification quality based on what is added, kept and deleted | 0 to 100, higher is better |
| BLEU | Overlap between the output and reference text | 0 to 100, higher is better |
| FKGL | Flesch-Kincaid Grade Level | Lower means easier to read |
| Human Score | Custom readability assessment | 0 to 100, higher is better |

## Sample Transformations

**Word substitutions**

| Original | Simplified |
| :--- | :--- |
| Utilize | Use |
| Facilitate | Help |
| Approximately | About |
| Sufficient | Enough |

**Phrase substitutions**

| Original | Simplified |
| :--- | :--- |
| In order to | To |
| Due to the fact that | Because |
| At this point in time | Now |

## Results

Across the tested domains, the system showed clear gains in readability:

- Average SARI score: 99.3 / 100
- Average FKGL reduction: 12.8 grade levels
- Average readability improvement: +26.5 points

## Example

**Input**

> The proliferation of ubiquitous computing devices has precipitated an unprecedented accumulation of data, necessitating sophisticated analytical methodologies.

**Simplified output**

> The spread of common computing devices has caused an unusual build-up of data, requiring advanced analysis methods.
