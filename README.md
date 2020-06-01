# SO / Kaggle Topic Modelling

1. `preprocessing`: Preprocess StackOverflow and Kaggle data
2. `hptuning`: Grid search over Mallet LDA model using Adelaide Uni's Phoenix HPC
3. `tsne`: Grid search t-sne hyperparameters using Phoenix HPC to find best looking TSNE plots
4. `analysis`: Perform the main analysis on the final data.
   1. `read_results.py` reads the output results of hptuning
   2. `find_best_topic_model.py` is a helper file for manual examination of LDA topics
   3. `main.py` produces all the plots in the paper




