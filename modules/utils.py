""" Supporting functions for main.py. """


def flatten_config(config: dict) -> dict:
    """Flatten configuration dictionary, except keys with 'kwargs' in them.

    Args:
        config (dict): configuration to flatten

    Returns:
        dict: flattened configuration
    """
    flat_config = {}
    for key, val in config.items():
        if isinstance(val, dict):
            if 'kwargs' in key:
                flat_config[key] = val
            else:
                flat_config.update(flatten_config(val))
        else:
            flat_config[key] = val
    return flat_config
