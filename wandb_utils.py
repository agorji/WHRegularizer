import wandb

def get_wandb_logs(wandb_path, wandb_config):
    api = wandb.Api()

    project_path = wandb_path.rsplit("/", 1)[0]
    filters = {f'config.{k}': v for k, v in wandb_config.items()}
    runs = api.runs(project_path, filters=filters)

    if len(runs) == 1:
        return runs[0].history(pandas=False)
    elif len(runs) == 0:
        return None
    else:
        raise Exception(f"{len(runs)} runs found with the given config, can't pick one.")