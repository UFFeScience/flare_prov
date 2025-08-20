#!/bin/bash

# CRIAR VARIAVEL versioning_control
versioning_control=false

# Executa o script Python e captura o caminho da worktree como última linha
WORKTREE_PATH=$(python3 utils/start_trial.py  \
                --versioning_control "$versioning_control" \
                | tail -n 1)

if $versioning_control; then
    # Muda para o diretório e abre shell interativo
    cd "$WORKTREE_PATH" && exec $SHELL
fi
