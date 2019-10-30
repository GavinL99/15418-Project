# Asynchronous Parallel Gradient Boosting using Parameter Server

## Summary
We plan on implementing a gradient boosting decision tree (GBDT) algorithm in an asynchronous framework. We are going to distribute the work in parallel through a parameter server, whilst first creating a proof of concept in OpenMP and MPI. Performance will be compared with sequential and parallel implementation in OpenCV and XGBoost.

## Background
Gradient boosting is a machine learning algorithm that ensembles weak learners like decision stumps and improves accuracy. The basic idea is to incrementally train a model to minimize the empirical loss function over the function space by fitting a weak learner that points in the negative gradient direction. We will also use decision trees in our implementation of gradient boosting.

Many researchers have proposed various ways to parallelize GBDT algorithm by generating good subsample of the original dataset and have worker nodes train weak learners, usually decision trees, on each subset. Explicit synchronization will be done at the end of every iteration to aggregate all trees built. However, this fork-join paradigm fails to scale as a small number of slow worker nodes can significantly slow down the training. For example, LightGBM, the state-of-art parallel GBDT framework, usually only achieve 5x to 7x speedup on a 32-core machines.

That's where the asynchronous parallel GBDT with parameter server framework comes to rescue: the server receives trees from workers, and workers build trees on subsamples of dataset asynchronously, which allows overlapping of comminucation and computation time.

