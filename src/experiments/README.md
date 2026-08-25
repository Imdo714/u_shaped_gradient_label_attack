# Experiment modules

```text
experiments/
|-- training/          victim training and gradient collection
|-- analysis/          attack metrics and epoch analysis
|-- clustering/        gradient/smashed clustering and anchor mapping
|-- inference/         single-image attack execution
|-- reconstruction/    decoder and holdout reconstruction experiments
|-- attacks/           reusable attack algorithms
```

The root-level `run_full_experiment` and reconstruction modules are stable
compatibility entry points. New implementation code belongs in the role-based
packages above.
