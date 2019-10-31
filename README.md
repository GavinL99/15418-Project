# Asynchronous Parallel Gradient Boosting using Parameter Server

## Summary
We plan on implementing a gradient boosting decision tree (GBDT) algorithm in an asynchronous framework. We are going to distribute the work in parallel through a parameter server, whilst first creating a proof of concept in OpenMP and MPI. Performance will be compared with sequential and parallel implementation in OpenCV and XGBoost.

## Background
Gradient boosting is a machine learning algorithm that ensembles weak learners like decision stumps and improves accuracy. The basic idea is to incrementally train a model to minimize the empirical loss function over the function space by fitting a weak learner that points in the negative gradient direction. We will also use decision trees in our implementation of gradient boosting.

Many researchers have proposed various ways to parallelize GBDT algorithm by generating good subsample of the original dataset and have worker nodes train weak learners, usually decision trees, on each subset. Explicit synchronization will be done at the end of every iteration to aggregate all trees built. However, this fork-join paradigm fails to scale as a small number of slow worker nodes can significantly slow down the training. For example, LightGBM, the state-of-art parallel GBDT framework, usually only achieve 5x to 7x speedup on a 32-core machines.

That's where the asynchronous parallel GBDT with parameter server framework comes to rescue: the server receives trees from workers, and workers build trees on subsamples of dataset asynchronously, which allows overlapping of comminucation and computation time.

## Challenge
There are two major challenges of implementing async-GBDT algorithm:
* How to build the parameter server for workers to pull and commit their work.
* How to ensure load balancing such that the server node itself will not become the bottleneck.

Our proposed solution is to have multiple server nodes instead of one and implement a shared work queue under the producer-consumer framework. Having multiple server node incurs more communication costs, so we need to experiment and do profiling to find the best configuration of server / worker nodes.

Additionally we need to tune every gradient boosting algorithm on the ratio of data to train and to test on. As a result, some algorithms may lose accuracy in order to remain faster or vice versa.

## Resources
There are various synchronous implementations of gradient boosting algorithm using fork-join parallel method. We refer to the following papers for efficient implementation of stochastic gradient boosting and delayed gradient descent (DC-ASGD): 
\\
\text{[1]} J. H. Friedman, “Stochastic gradient boosting,” Computational Statistics Data Analysis, vol. 38, no. 4, pp. 367–378, 2002.

\text{[2]} J. Ye, J. H. Chow, J. Chen, and Z. Zheng, “Stochastic gradient boosted distributed decision trees,” in Acm Conference on Information Knowl- edge Management, 2009, pp. 2061–2064.
\\
We are going to draw inspiration for the base algorithm from the following article by Rory Mitchell describing a gradient boosting implementation through CUDA:
\\
\text{[3]} \href{https://devblogs.nvidia.com/gradient-boosting-decision-trees-xgboost-cuda/}{Gradient Boosting, Decision Trees and XGBoost with CUDA, Nvidia}
\\
From that article, we will also use the XGBoost implementation outlined to compare our results with the current standard implementation.

We mainly refer to the following paper for asynchronous implementation of GBDT and We plan to follow the parameter server framework outlined there. 
\\
\text{[4]} \href{https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf}{Daning, Fen, et al. "Asynch-SGBDT: Train a Stochastic Gradient Boosting Decision Tree in an Asynchronous Parallel Manner"}
