## 1. Python venv
* create venv: python3 -m venv env_HMRs.
* use venv: source env_HMRs/bin/activate
## 2. Install jupyter-notebook in venv
* python3 -m pip install notebook
* python3 -m pip install jupyter
* python3 -m ipykernel install --user --name="code_HMRs" --display-name="code_HMRs"

## 3. Start a local Ray instance
```
   ray.init(
        #_temp_dir=str(PathUtils.scratch_dir / "ray"),
        ignore_reinit_error=True,
        local_mode=local_mode,
    )
```

## 4. Start Tensorboard
* tensorboard --logdir . --host 0.0.0.0 --port 6006
