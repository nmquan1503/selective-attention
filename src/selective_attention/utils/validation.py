def check_required(
    obj, 
    name: str,
    condition: bool,
    stage: str
):
    if condition and obj is None:
        raise ValueError(f"{name} must be provided during {stage}.")
