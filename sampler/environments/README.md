# Environments

This folder contains environment implementations used by `L1Expander`.

To add a new environment:

1. Create a new class in this folder.
2. Inherit from `BaseEnvironment`.
3. Set a unique `env_type` string on the class.
4. Implement:

```python
def run(self, request: EnvironmentRequestBundle) -> EnvironmentFeedbackBundle:
    ...
```

5. Return an `EnvironmentFeedbackBundle`.
6. Export the class in [__init__.py](/home/shuyi/BOA/sampler/environments/__init__.py).
7. Register the class in [l1_expander.py](/home/shuyi/BOA/sampler/l1_expander.py) inside `self.environments`.

Current examples:

- `SingleTurnEnvironment`

Expected behavior:

- `L1Expander` routes by `request.env_type`
- the environment runs
- it returns an `EnvironmentFeedbackBundle`
- `L1Expander` attaches returned feedback sequences to the tree
