## Setup

```shell
poetry install
```

## Starting Training

```shell
poetry run python -m fib.main
```

You should see something like

```
Loss: 3.3556536047458647
Loss: 2.4866632815599443
Loss: 2.0669580459594727
3456it [00:04, 722.73it/s]
```

## Tracing a Single Metric

```shell
poetry run viztracer --include_files "tests" -m fib.main
```

The exported JSON should only have a single metric logged
without extensive tracing.