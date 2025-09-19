# Code Language Models for Security Patch Management: How Far Are We?

This repository contains the replication package for the paper **"Code Language Models for Security Patch Management: How Far Are We?"**. The study We conduct the first large-scale empirical study on 
fine-tuning or prompting nine state-of-the-art CodeLMs for three security-patch-related downstream tasks, including silent patch identification, record-patch linking, and vulnerability description generation, 
covering classification, ranking, and generation problems.


## Abstract

The rapid expansion of open-source software has also brought significant security challenges, particularly introducing and propagating security vulnerabilities. 
In response, effective security patch management establishes a continuous, structured pipeline by systematically identifying, testing, and deploying security patches to fix vulnerabilities. 
However, manually managing a large number of security patches (i.e., any update is approved and installed by hand) is time-consuming, leading to a great motivation for automating this process. 
Although Code Language Models (CodeLMs) have shown potential in various code-centric tasks, there remains an open question as to how well CodeLMs perform within the context of security patch management.

To bridge this gap, we performed the first comprehensive empirical study on fine-tuning or prompting nine state-of-the-art CodeLMs for three security-patch-related downstream tasks, 
including silent patch identification (distinguishing security patches from normal commits), record-patch linking (connecting authoritative vulnerability records, e.g., CVE, to the corresponding fixing commits), 
and vulnerability description generation (providing a piece of text summarizing the vulnerability fixed by the patch), covering classification, ranking, and generation problems. 
Our findings reveal that there is no ``one-size-fits-all'' model that can always perform the best. 
Furthermore, directly prompting LLMs to solve security-patch-related tasks is impractical due to the lack of task-specific knowledge. 
Additionally, existing automated evaluation metrics cannot fully reflect the capability of LLMs in considered tasks. 
These findings underscore the considerable gap between current capabilities and the practical requirements for deploying CodeLMs in automating security patch management.

## Project Structure

The repository contains:
- Datasets: the fine-tuning dataset for three security patch management tasks
- task-1: the scripts for task-1
- task-2: the scripts for task-2
- task-3: the scripts for task-3

## Research Questions
1. **RQ1:** How effective are fine-tuned CodeLMs for security patch management?
2. **RQ2:** How do fine-tuning approaches perform compared to prompt engineering approaches?


## How to Use
1. **Clone the Repository:**
```
git clone https://github.com/LLM4SPM/LLM4SPM.git
```
2. **Run Experiments:**
Navigate to the `task-[1-3]/Fine-Tune` and `task-[1-3]/Prompt` directory and follow the instructions in the respective subdirectories to reproduce the experiments.
3. **Results:**
Navigate to the `task-[1-3]/Results` directory to evaluate the results.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the developers of the tools and datasets used in this study. Special thanks to the reviewers for their valuable feedback.
