#!/bin/bash
set -e

setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
        fi
        mkdir -p /run/sshd
        /usr/sbin/sshd
        echo "SSH server is running."
    else
        echo "No PUBLIC_KEY env var. SSH disabled."
    fi
}

echo "Preparing model..."
python3 /app/src/utils.py

echo "Reading model path..."
LOCAL_MODEL_PATH=$(cat /app/.model_path)
export LOCAL_MODEL_PATH

if [ "$MODE" == "interactive" ]; then
    echo "Interactive mode. Keeping container alive..."
    sleep infinity
else
    echo "Starting RunPod handler..."
    python3 /app/src/handler.py
fi