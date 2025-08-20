import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
import shutil

def carregar_info_trial() -> dict:
    info_path = Path("trial_info.json")
    if not info_path.exists():
        raise FileNotFoundError(
            "trial_info.json não encontrado no diretório atual."
        )

    with open(info_path, "r") as f:
        return json.load(f)


def calcular_tempo_decorrido(timestamp_inicio: str) -> str:
    inicio = datetime.strptime(timestamp_inicio, "%Y%m%dT%H%M%S")
    agora = datetime.utcnow()
    delta = agora - inicio
    return str(delta)


def criar_commit_final(duracao: str, hash_valor: str):
    try:
        with open("trial_duration.txt", "w") as f:
            f.write(f"Duração total: {duracao}\n")

        subprocess.run(["git", "add", "."], check=True)
        msg = f"Finalização do trial - HASH: {hash_valor} - duração: {duracao}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        print("Commit final criado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando Git: {e}")
        raise

def copiar_e_commit_modelo(hash_valor: str):
    try:
        # Define source and destination paths
        source_path = os.path.join("workspace", "fed_clustering", "prod_01", "server1", "kmeans_model.pkl")
        dest_path = "./kmeans_model.pkl"

        # Copy the file
        shutil.copyfile(source_path, dest_path)
        print(f"Modelo copiado de '{source_path}' para '{dest_path}'.")

        # Git add and commit
        subprocess.run(["git", "add", "kmeans_model.pkl"], check=True)
        msg = f"Salvando modelo KMeans - HASH: {hash_valor}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        print("Commit do modelo criado com sucesso.")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {source_path}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar comando Git: {e}")
        raise

if __name__ == "__main__":
    try:
        info = carregar_info_trial()
        duracao = calcular_tempo_decorrido(info["timestamp"])
        copiar_e_commit_modelo(
            info["hash_trial"]
        )
        criar_commit_final(duracao, info["hash_trial"])
        print(f"trial finalizado com duração: {duracao}")

    except Exception as e:
        print(f"Erro ao finalizar trial: {e}")
