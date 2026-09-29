"""
Push training outputs to the Hugging Face Hub
"""
import os

from huggingface_hub import HfApi


def push_to_hub(path: str) -> None:
    """
    Upload a checkpoint or results file to the Hub repo named by $HF_REPO (if set),
    at the same relative path, e.g., checkpoints/<model_config>/<run_name>_distill.pt
    """
    repo_id = os.environ.get('HF_REPO')
    if not repo_id:
        return
    try:
        api = HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=os.path.relpath(path),
                        repo_id=repo_id, commit_message=f'Upload {os.path.basename(path)}')
        print(f'-> Pushed {path} to https://huggingface.co/{repo_id}')
    except Exception as e:  # Keep training if an upload fails
        print(f'-> Failed to push {path} to {repo_id}: {e}')
