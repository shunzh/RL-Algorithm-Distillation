def find_config_file(env_id: str, alg: str):
    if env_id.startswith('DarkRoom'):
        return f"configs/DarkRoom-{alg}.json"
    else:
        raise ValueError(f"Unknown environment {env_id} or algorithm {alg}")
