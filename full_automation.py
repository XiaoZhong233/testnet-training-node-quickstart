import json
import os

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

HF_USERNAME = os.environ["HF_USERNAME"]

if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    # load trainin args
    # define the path of the current file
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)

    task = get_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url
    data_url = task["data"]["training_set_url"]
    # 使用公开的数据集
    # data_url = "https://cdn-lfs.hf.co/repos/0e/c9/0ec907794f5c29d64b130fef8309613dd2efe6eeb9d92c363f0a30e40caa788a/aab86ee50d2e8fabd3a2c4b9f82bdff2dbadd573cc15a85bd96ee47decc01ba8?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27stanford_alpaca_data.jsonl%3B+filename%3D%22stanford_alpaca_data.jsonl%22%3B&Expires=1736839247&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNjgzOTI0N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy8wZS9jOS8wZWM5MDc3OTRmNWMyOWQ2NGIxMzBmZWY4MzA5NjEzZGQyZWZlNmVlYjlkOTJjMzYzZjBhMzBlNDBjYWE3ODhhL2FhYjg2ZWU1MGQyZThmYWJkM2EyYzRiOWY4MmJkZmYyZGJhZGQ1NzNjYzE1YTg1YmQ5NmVlNDdkZWNjMDFiYTg%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=uXjy2y-mjjoNTuv1dGm3G9uERJyZdWY0tPSSh4QNlUtI-R4tziTJaqe557AdMhhVo%7EeqBZhUqbHmLGLMMw%7EXjoHRP8SNrLV8ucGavzadeSn-aEYXLGpEUCo5JbKSa-ZtGDSxtU-LW7Akvg14cOb%7EpJEwGHT2QDnFgPzBlvyeSMy4iGzHKyldlh%7EHHEtLFye8rS9-JdO%7EsyVC8Y75k10j1jOVXw5dNuySpCf9R2Y1A5HKAQjTeysNNKE1Ft7XtfXtt0siuiTiD0c%7E6EkUtYLCLTj7gS5DgecPSFxdDApqAeXLNe4HKbHRc5kfa84v-9hmJad0jmrqgORvrJQbdfJRqA__&Key-Pair-Id=K3RPWS32NSSJCE"
    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    # filter out the model within the max_params
    model2size = {k: v for k, v in model2size.items() if v <= max_params}
    all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")
    # download in chunks
    response = requests.get(data_url, stream=True)
    with open("data/demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # train all feasible models and merge
    for model_id in all_training_args.keys():
        logger.info(f"Start to train the model {model_id}...")
        # if OOM, proceed to the next model
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
            )
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue

        # generate a random repo id based on timestamp
        gpu_type = get_gpu_type()

        try:
            logger.info("Start to push the lora weight to the hub...")
            api = HfApi(token=os.environ["HF_TOKEN"])
            repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
            # check whether the repo exists
            try:
                api.create_repo(
                    repo_name,
                    exist_ok=False,
                    repo_type="model",
                )
            except Exception:
                logger.info(
                    f"Repo {repo_name} already exists. Will commit the new version."
                )

            commit_message = api.upload_folder(
                folder_path="outputs",
                repo_id=repo_name,
                repo_type="model",
            )
            # get commit hash
            commit_hash = commit_message.oid
            logger.info(f"Commit hash: {commit_hash}")
            logger.info(f"Repo name: {repo_name}")
            # submit
            submit_task(
                task_id, repo_name, model2base_model[model_id], gpu_type, commit_hash
            )
            logger.info("Task submitted successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
        finally:
            # cleanup merged_model and output
            os.system("rm -rf merged_model")
            os.system("rm -rf outputs")
            continue
