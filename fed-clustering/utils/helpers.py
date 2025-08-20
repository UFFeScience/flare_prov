import json


def get_hash_trial(file_path="./trial_info.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["hash_trial"]
