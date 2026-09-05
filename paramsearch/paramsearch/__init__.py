"""Parameter search framework for the locust simulation.

Stages: Sobol screening (paramsearch.screening) narrows ~30 parameters to the
sensitive subset, then Bayesian optimization (paramsearch.optimize) calibrates
that subset against field-data targets (paramsearch.objective). Both stages
share one evaluation path (paramsearch.evaluation) that runs the Scala
simulation headless and reduces its binary snapshots to band metrics
(paramsearch.metrics).
"""
