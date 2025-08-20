import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import os
import json
import argparse

CONFIG_PATHS = [
    "./jobs/sklearn_kmeans_base/app/config/config_fed_server.json",
    "./jobs/sklearn_kmeans_base/app/config/config_fed_client.json",
]

WORKTREE_DIR = Path("./trials")  # onde as worktrees vão ficar


def gerar_hash(conteudo: str, usuario_git: str, timestamp: str) -> str:
    dados = (
        conteudo.encode("utf-8")
        + usuario_git.encode("utf-8")
        + timestamp.encode("utf-8")
    )
    return hashlib.sha1(dados).hexdigest()


def obter_usuario_git() -> str:
    resultado = subprocess.run(
        ["git", "config", "user.name"],
        capture_output=True,
        text=True,
        check=True,
    )
    return resultado.stdout.strip().replace(" ", "_").lower()


def ler_conteudo_arquivos() -> str:
    conteudos = ""
    for caminho in CONFIG_PATHS:
        try:
            with open(caminho, "rb") as f:
                conteudos += f.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Erro ao ler {caminho}: {e}")
    return conteudos


def criar_branch_e_checkout(usuario: str, timestamp: str) -> Path:
    branch_name = f"trials/{usuario}_{timestamp}"
    worktree_path = WORKTREE_DIR / f"{usuario}_{timestamp}"

    WORKTREE_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "branch", branch_name], check=True)
    subprocess.run(
        ["git", "worktree", "add", str(worktree_path), branch_name], check=True
    )

    print(f"Branch criada: {branch_name}")
    print(f"Worktree criada em: {worktree_path}")

    return worktree_path


def salvar_info_local(
    worktree_path: Path, hash_valor: str, usuario: str, timestamp: str
):
    info = {"usuario": usuario, "timestamp": timestamp, "hash_trial": hash_valor}
    print(worktree_path)
    info_path = worktree_path / "fed-clustering/trial_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Arquivo de info salvo em: {info_path}")


def criar_commit_no_trial(worktree_path: Path, hash_valor: str):
    os.chdir(worktree_path)
    try:
        subprocess.run(["git", "add", "."], check=True)
        mensagem_commit = f"Commit inicial do trial - HASH: {hash_valor}"
        subprocess.run(["git", "commit", "-m", mensagem_commit], check=True)
        print("Commit criado com sucesso.")

    finally:
        os.chdir("..")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versioning_control",
        type=str2bool,
        default=False,
    )
    args = parser.parse_args()
    versioning_control = args.versioning_control
    usuario = obter_usuario_git()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    conteudo_config = ler_conteudo_arquivos()
    hash_exp = gerar_hash(conteudo_config, usuario, timestamp)

    print(f"Hash do trial: {hash_exp}")
    if versioning_control:
        path_trial = criar_branch_e_checkout(usuario, timestamp)
        salvar_info_local(path_trial, hash_exp, usuario, timestamp)
        criar_commit_no_trial(path_trial, hash_exp)
        print(f"\nAbra o terminal na worktree com:\ncd {path_trial}\n")
        print(f"Trial {hash_exp} criado com sucesso.")
        print(path_trial)
    else:
        salvar_info_local(Path("../"), hash_exp, usuario, timestamp)

if __name__ == "__main__":
   main()

