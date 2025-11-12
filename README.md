This repo contains the code and data associated with paper [**Optimized Distortion in Linear Social Choice**](https://arxiv.org/abs/2510.20020) accepted by AAAI 2026 for oral presentation. 


## Requirements
We use gurobi for optimization. You can also try other software. We chose it becauce (1) a free student license can be obtained on their website (2) they support warm starting, which is a very helpful function for our problem as many times we have optimization problems with same constraints but different objectives.

For experiemnts without a need for language embeddings:
```bash
pip install numpy scipy gurobipy sklearn pandas 
```
For experiments with a need of language embeddings like those on [the abortion opinion dataset](https://github.com/generative-social-choice/survey_data/), we used:
```bash
pip install sentence_transformers
```
## Background

The repository is organized around the two main experiments described in the paper.

The core implementation is contained in `main.py`, which defines the class `Instance`.  
This class provides the central functionality for encoding the problem setup.  
Its method `distortion_comparisons()` computes various social choice rules and their corresponding distortions.  
For simpler rules such as **Plurality** and **Borda**, this method calls the helper function `calculate_distortion()`, also defined within the class.  
Although these "easy" rules are straightforward to implement, their **theoretical distortion bounds** must be solved programmatically, as implemented in the `calculate_theoretical_distortion()` function.

More complex rules are also included:
1. **LSLR**, which uses `uporj` and `stable_lottery` (implemented separately in `stable_lottery.py`), and  
2. **Instance-optimal** and **deterministic** rules, implemented in `calculate_optimal_random()` and `calculate_optimal_det()`.

### MovieLens Data
For the MovieLens experiments, we sample a user–movie interaction matrix from the larger dataset and construct the “ground-truth” utility matrix through a combination of matrix factorization , renormalization and learning.  
The preprocessing and experiment-running code for this setup is found in `experiment.py` and `main.py`.

### Abortion Opinion Data
For the Abortion Opinion experiments, we first apply sentence embeddings to the textual statements.  
As we vary the embedding dimension `d`, we perform PCA on the embeddings before running the experiments.  
The preprocessing and experiment-running code for this setup is likewise located in `experiment.py` and `main.py`.






## Citations
If you use this code, please cite:
 ```
 @inproceeding{ge2025optimizeddistortionlinearsocial,
        title={Optimized Distortion in Linear Social Choice}, 
        author={Luise Ge and Gregory Kehne and Yevgeniy Vorobeychik},
        year={2026},
        booktitle={AAAI}
  }
```


