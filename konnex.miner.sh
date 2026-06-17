#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/vsmelov/knx-subnet-drone-navigation.git"
IMAGE_NAME="xsubnet-drone-miner:local"
COMPOSE_FILE=".docker-compose.miner.installer.yml"

DEFAULT_INSTALL_DIR="${KONNEX_INSTALL_DIR:-$HOME/konnex-miner}"
DEFAULT_SUBTENSOR_CHAIN_ENDPOINT="wss://testnet-rpc1.konnex.world:39944"
DEFAULT_NETUID="4"
DEFAULT_MINER_AXON_PORT="8091"
DEFAULT_OPENAI_MODEL="gpt-5.4"
DEFAULT_OPENFLY_HF_MODEL="IPEC-COMMUNITY/openfly-agent-7b"
DEFAULT_OPENFLY_LOCAL_MODEL="/app/models/openfly-agent-7b"
DEFAULT_WALLET_BALANCE_TIMEOUT_SECONDS="${KONNEX_WALLET_BALANCE_TIMEOUT_SECONDS:-30}"

OPENAI_MODELS=(
  "gpt-5.4-mini"
  "gpt-5.4"
  "gpt-5.5"
)

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
info() { printf '[info] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
die() { printf '[error] %s\n' "$*" >&2; exit 1; }

ask() {
  local prompt="$1" default="${2:-}" value
  if [[ -n "$default" ]]; then
    printf '\033[1;36m%s\033[0m [%s]: ' "$prompt" "$default" >&2
    read -r value
    printf '%s' "${value:-$default}"
  else
    printf '\033[1;36m%s\033[0m: ' "$prompt" >&2
    read -r value
    printf '%s' "$value"
  fi
}

ask_secret() {
  local value
  printf '\033[1;36m%s\033[0m: ' "$1" >&2
  read -r -s value
  printf '\n' >&2
  printf '%s' "$value"
}

ask_yes_no() {
  local prompt="$1" default="${2:-n}" answer
  while true; do
    printf '\033[1;36m%s\033[0m [y/n]: ' "$prompt"
    read -r answer
    answer="${answer:-$default}"
    case "$answer" in
      y|Y|yes|YES|Yes) return 0 ;;
      n|N|no|NO|No) return 1 ;;
      *) warn "Type y or n." ;;
    esac
  done
}

need_sudo() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

ensure_ubuntu_host() {
  [[ -r /etc/os-release ]] || die "This installer is intended for Ubuntu."
  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || die "This installer is intended for Ubuntu, detected: ${PRETTY_NAME:-unknown}."

  if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
    die "Run this on a real VPS/bare-metal Ubuntu host, not inside a container."
  fi
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    printf 'docker'
  elif command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    printf 'sudo docker'
  else
    return 1
  fi
}

dkr() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif need_sudo docker info >/dev/null 2>&1; then
    need_sudo docker "$@"
  else
    start_docker || die "Docker daemon is not running. Start Docker and rerun this script."
    if docker info >/dev/null 2>&1; then
      docker "$@"
    else
      need_sudo docker "$@"
    fi
  fi
}

dkr_timeout() {
  local seconds="$1"
  shift
  if ! command -v timeout >/dev/null 2>&1; then
    dkr "$@"
  elif docker info >/dev/null 2>&1; then
    timeout --foreground "${seconds}s" docker "$@"
  else
    timeout --foreground "${seconds}s" sudo docker "$@"
  fi
}

start_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    need_sudo systemctl enable --now docker
  elif command -v service >/dev/null 2>&1; then
    need_sudo service docker start || true
  else
    return 1
  fi

  local i
  for ((i = 1; i <= 30; i++)); do
    docker info >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

require_docker() {
  docker info >/dev/null 2>&1 || need_sudo docker info >/dev/null 2>&1 || start_docker || \
    die "Docker daemon is not running. Start Docker and rerun this script."
}

install_base_packages() {
  need_sudo apt-get update
  need_sudo apt-get install -y ca-certificates curl gnupg git lsb-release
}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    info "Docker and Docker Compose are already installed."
    require_docker
    return
  fi

  install_base_packages
  bold "Installing Docker"
  need_sudo install -m 0755 -d /etc/apt/keyrings

  if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | need_sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
    need_sudo chmod a+r /etc/apt/keyrings/docker.gpg
  fi

  local codename arch
  codename="$(. /etc/os-release && printf '%s' "$VERSION_CODENAME")"
  arch="$(dpkg --print-architecture)"
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu %s stable\n' "$arch" "$codename" \
    | need_sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  need_sudo apt-get update
  need_sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  require_docker

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    need_sudo usermod -aG docker "$USER" || true
    warn "User '$USER' was added to the docker group. If Docker permission fails, log out and back in."
  fi
}

docker_gpu_runtime_works() {
  dkr run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1
}

install_nvidia_toolkit() {
  command -v nvidia-smi >/dev/null 2>&1 || \
    die "NVIDIA GPU/driver was not detected. GPU/OpenFly mode needs an NVIDIA GPU host."

  if docker_gpu_runtime_works; then
    info "Docker NVIDIA runtime is already working."
    return
  fi

  bold "Installing NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | need_sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | need_sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

  need_sudo apt-get update
  need_sudo apt-get install -y nvidia-container-toolkit
  need_sudo nvidia-ctk runtime configure --runtime=docker

  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    need_sudo systemctl restart docker
  elif command -v service >/dev/null 2>&1; then
    need_sudo service docker restart || true
  fi

  docker_gpu_runtime_works || warn "Docker GPU runtime is not active yet. Restart Docker or the server, then rerun start."
}

prepare_repo() {
  info "Install directory: $DEFAULT_INSTALL_DIR"
  mkdir -p "$DEFAULT_INSTALL_DIR"
  cd "$DEFAULT_INSTALL_DIR"

  if [[ -d .git ]]; then
    git fetch --all --prune >/dev/null 2>&1 || true
  elif [[ -z "$(find . -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    git clone "$REPO_URL" .
  else
    die "Install directory is not empty and is not a git repo: $DEFAULT_INSTALL_DIR"
  fi

  git submodule update --init --recursive
  mkdir -p wallets logs logs/miner-hf-cache logs/miner-torch-cache models
}

load_install() {
  ensure_ubuntu_host
  [[ -d "$DEFAULT_INSTALL_DIR/.git" ]] || die "Install not found at $DEFAULT_INSTALL_DIR. Run install first."
  cd "$DEFAULT_INSTALL_DIR"
  load_env
  set_defaults
}

load_env() {
  [[ -f .env ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" && "$line" != \#* && "$line" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    value="${value%$'\r'}"
    case "$key" in
      NETUID|SUBTENSOR_CHAIN_ENDPOINT|MINER_WALLET_NAME|MINER_WALLET_HOTKEY|MINER_AXON_PORT|MINER_EXTERNAL_IP|MINER_EXTERNAL_PORT|OPENFLY_SUBNET_MINER_MODEL|OPENFLY_MODEL|OPENFLY_ATTN_IMPLEMENTATION|HF_TOKEN|OPENAI_API_TOKEN|OPENFLY_SUBNET_MINER_OPENAI_MODEL)
        export "$key=$value"
        ;;
    esac
  done < .env
}

set_defaults() {
  export NETUID="${NETUID:-$DEFAULT_NETUID}"
  export SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-$DEFAULT_SUBTENSOR_CHAIN_ENDPOINT}"
  export MINER_WALLET_NAME="${MINER_WALLET_NAME:-miner}"
  export MINER_WALLET_HOTKEY="${MINER_WALLET_HOTKEY:-default}"
  export MINER_AXON_PORT="${MINER_AXON_PORT:-$DEFAULT_MINER_AXON_PORT}"
  export MINER_EXTERNAL_IP="${MINER_EXTERNAL_IP:-}"
  export MINER_EXTERNAL_PORT="${MINER_EXTERNAL_PORT:-$MINER_AXON_PORT}"
  export OPENFLY_SUBNET_MINER_MODEL="${OPENFLY_SUBNET_MINER_MODEL:-openai}"
  export OPENFLY_MODEL="${OPENFLY_MODEL:-$DEFAULT_OPENFLY_LOCAL_MODEL}"
  export OPENFLY_ATTN_IMPLEMENTATION="${OPENFLY_ATTN_IMPLEMENTATION:-eager}"
  export HF_TOKEN="${HF_TOKEN:-}"
  export OPENAI_API_TOKEN="${OPENAI_API_TOKEN:-}"
  export OPENFLY_SUBNET_MINER_OPENAI_MODEL="${OPENFLY_SUBNET_MINER_OPENAI_MODEL:-$DEFAULT_OPENAI_MODEL}"
}

write_env() {
  cat > .env <<EOF
NETUID=$NETUID
SUBTENSOR_CHAIN_ENDPOINT=$SUBTENSOR_CHAIN_ENDPOINT
MINER_WALLET_NAME=$MINER_WALLET_NAME
MINER_WALLET_HOTKEY=$MINER_WALLET_HOTKEY
MINER_AXON_PORT=$MINER_AXON_PORT
MINER_EXTERNAL_IP=$MINER_EXTERNAL_IP
MINER_EXTERNAL_PORT=$MINER_EXTERNAL_PORT
OPENFLY_SUBNET_MINER_MODEL=$OPENFLY_SUBNET_MINER_MODEL
OPENFLY_MODEL=$OPENFLY_MODEL
OPENFLY_ATTN_IMPLEMENTATION=$OPENFLY_ATTN_IMPLEMENTATION
HF_TOKEN=$HF_TOKEN
OPENAI_API_TOKEN=$OPENAI_API_TOKEN
OPENFLY_SUBNET_MINER_OPENAI_MODEL=$OPENFLY_SUBNET_MINER_OPENAI_MODEL
EOF
  chmod 600 .env
}

set_env_value() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp)"
  if [[ -f .env ]]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { found = 0 }
      index($0, key "=") == 1 { print key "=" value; found = 1; next }
      { print }
      END { if (!found) print key "=" value }
    ' .env > "$tmp"
  else
    printf '%s=%s\n' "$key" "$value" > "$tmp"
  fi
  mv "$tmp" .env
  chmod 600 .env
}

detect_public_ip() {
  local url ip
  for url in https://api.ipify.org https://ifconfig.me/ip https://icanhazip.com; do
    ip="$(curl -fsS4 --max-time 5 "$url" 2>/dev/null | tr -d '[:space:]' || true)"
    [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && { printf '%s' "$ip"; return 0; }
  done
  return 1
}

ensure_external_ip() {
  [[ -n "${MINER_EXTERNAL_IP:-}" ]] && return
  MINER_EXTERNAL_IP="$(detect_public_ip || true)"
  if [[ -n "$MINER_EXTERNAL_IP" ]]; then
    info "Detected external axon IP: $MINER_EXTERNAL_IP"
  else
    warn "Could not detect external IP. Set MINER_EXTERNAL_IP in .env if validators cannot reach the miner."
  fi
}

open_axon_port() {
  if command -v ufw >/dev/null 2>&1 && need_sudo ufw status 2>/dev/null | grep -qi '^Status: active'; then
    need_sudo ufw allow "${MINER_AXON_PORT}/tcp" >/dev/null || true
    info "Allowed TCP port $MINER_AXON_PORT in ufw."
  fi
}

check_axon_port() {
  local published external_port
  external_port="${MINER_EXTERNAL_PORT:-$MINER_AXON_PORT}"
  published="$(dkr compose -f "$COMPOSE_FILE" port subnet-miner "$MINER_AXON_PORT" 2>/dev/null || true)"
  [[ -n "$published" ]] && info "Docker published axon port: $published"
  [[ -n "${MINER_EXTERNAL_IP:-}" ]] && info "Announced external axon: $MINER_EXTERNAL_IP:$external_port"
}

check_openai_key() {
  local token="$1" tmp http_code
  tmp="$(mktemp)"
  http_code="$(
    curl -sS -o "$tmp" -w "%{http_code}" \
      -H "Authorization: Bearer $token" \
      https://codex.sale/v1/models || true
  )"
  rm -f "$tmp"
  [[ "$http_code" == "200" ]]
}

choose_openai_model() {
  local choice i current
  current="${OPENFLY_SUBNET_MINER_OPENAI_MODEL:-$DEFAULT_OPENAI_MODEL}"
  printf '\nOpenAI model, current: %s\n' "$current"
  for i in "${!OPENAI_MODELS[@]}"; do
    printf '  %s) %s\n' "$((i + 1))" "${OPENAI_MODELS[$i]}"
  done
  printf 'Model [Enter = current]: '
  read -r choice
  [[ -z "$choice" ]] && return
  [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#OPENAI_MODELS[@]} )) || die "Invalid model choice."
  OPENFLY_SUBNET_MINER_OPENAI_MODEL="${OPENAI_MODELS[$((choice - 1))]}"
}

prepare_openai_backend() {
  local token
  OPENFLY_SUBNET_MINER_MODEL="openai"
  set_env_value "OPENFLY_SUBNET_MINER_MODEL" "$OPENFLY_SUBNET_MINER_MODEL"

  token="${OPENAI_API_TOKEN:-}"
  if [[ -n "$token" ]]; then
    info "Checking saved OpenAI key..."
    if check_openai_key "$token"; then
      green "Saved OpenAI key is valid."
    else
      warn "Saved OpenAI key is invalid."
      token=""
    fi
  fi

  while [[ -z "$token" ]]; do
    token="$(ask_secret "OpenAI API key (Enter = no key)")"
    if [[ -z "$token" ]]; then
      warn "No OpenAI key. Miner will use fallback behavior if the code supports it."
      break
    fi
    check_openai_key "$token" && break
    warn "OpenAI key is invalid or has no API access."
    token=""
  done

  OPENAI_API_TOKEN="$token"
  choose_openai_model
  set_env_value "OPENAI_API_TOKEN" "$OPENAI_API_TOKEN"
  set_env_value "OPENFLY_SUBNET_MINER_OPENAI_MODEL" "$OPENFLY_SUBNET_MINER_OPENAI_MODEL"
  green "Using OpenAI backend: $OPENFLY_SUBNET_MINER_OPENAI_MODEL"
}

prepare_openfly_backend() {
  install_nvidia_toolkit
  OPENFLY_SUBNET_MINER_MODEL="openfly"
  set_env_value "OPENFLY_SUBNET_MINER_MODEL" "$OPENFLY_SUBNET_MINER_MODEL"

  [[ -n "${HF_TOKEN:-}" ]] || HF_TOKEN="$(ask_secret "HF token (Enter = no token)")"
  OPENFLY_MODEL="$(ask "OpenFly model" "${OPENFLY_MODEL:-$DEFAULT_OPENFLY_LOCAL_MODEL}")"
  OPENFLY_ATTN_IMPLEMENTATION="${OPENFLY_ATTN_IMPLEMENTATION:-eager}"

  set_env_value "HF_TOKEN" "$HF_TOKEN"
  set_env_value "OPENFLY_MODEL" "$OPENFLY_MODEL"
  set_env_value "OPENFLY_ATTN_IMPLEMENTATION" "$OPENFLY_ATTN_IMPLEMENTATION"
  ensure_openfly_model
  green "Using GPU/OpenFly backend."
}

ensure_openfly_model() {
  local host_model_dir="models/openfly-agent-7b"
  case "$OPENFLY_MODEL" in
    "$DEFAULT_OPENFLY_LOCAL_MODEL"|/app/models/openfly-agent-7b)
      if [[ -s "$host_model_dir/config.json" ]] || [[ -s "$host_model_dir/model.safetensors.index.json" ]]; then
        info "OpenFly model exists at $(pwd)/$host_model_dir."
        return
      fi
      bold "Downloading OpenFly model weights"
      warn "This downloads about 15 GB into $(pwd)/$host_model_dir."
      mkdir -p models logs/miner-hf-cache
      dkr run --rm \
        --entrypoint python3 \
        -e HF_TOKEN="$HF_TOKEN" \
        -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
        -v "$PWD/models:/app/models:rw" \
        -v "$PWD/logs/miner-hf-cache:/app/.cache/huggingface:rw" \
        "$IMAGE_NAME" \
        -c '
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
snapshot_download(
    repo_id="IPEC-COMMUNITY/openfly-agent-7b",
    local_dir="/app/models/openfly-agent-7b",
    token=token,
)
print("OpenFly model downloaded.")
'
      ;;
    IPEC-COMMUNITY/openfly-agent-7b)
      info "Using Hugging Face model id directly: $OPENFLY_MODEL"
      ;;
    *)
      info "Using custom OpenFly model: $OPENFLY_MODEL"
      ;;
  esac
}

choose_backend() {
  local choice current
  current="${OPENFLY_SUBNET_MINER_MODEL:-openai}"
  printf '\nMiner backend, current: %s\n' "$current"
  printf '  1) OpenAI API\n'
  printf '  2) GPU OpenFly\n'
  printf '> '
  read -r choice
  choice="${choice:-current}"
  case "$choice" in
    1|openai|api|OpenAI) prepare_openai_backend ;;
    2|gpu|openfly|OpenFly|GPU) prepare_openfly_backend ;;
    current)
      [[ "$current" == "openfly" ]] && prepare_openfly_backend || prepare_openai_backend
      ;;
    *) die "Invalid backend choice." ;;
  esac
}

write_compose() {
  local gpu_block="" external_axon_args=""
  [[ "$OPENFLY_SUBNET_MINER_MODEL" == "openfly" ]] && gpu_block="    gpus: all"
  if [[ -n "${MINER_EXTERNAL_IP:-}" ]]; then
    external_axon_args='      - --axon.external_ip
      - "${MINER_EXTERNAL_IP:-}"
      - --axon.external_port
      - "${MINER_EXTERNAL_PORT:-8091}"'
  fi

  cat > "$COMPOSE_FILE" <<EOF
services:
  subnet-miner:
    build:
      context: .
      dockerfile: docker/subnet-miner/Dockerfile
    image: $IMAGE_NAME
    restart: unless-stopped
$gpu_block
    env_file:
      - .env
    environment:
      NETUID: "\${NETUID:-1}"
      SUBTENSOR_CHAIN_ENDPOINT: "\${SUBTENSOR_CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
      WALLET_NAME: "\${MINER_WALLET_NAME:-miner}"
      WALLET_HOTKEY: "\${MINER_WALLET_HOTKEY:-default}"
      MINER_AXON_PORT: "\${MINER_AXON_PORT:-8091}"
      MINER_EXTERNAL_IP: "\${MINER_EXTERNAL_IP:-}"
      MINER_EXTERNAL_PORT: "\${MINER_EXTERNAL_PORT:-8091}"
      OPENAI_BASE_URL: "https://codex.sale/v1"
      OPENAI_API_BASE: "https://codex.sale/v1"
      OPENFLY_MODEL: "\${OPENFLY_MODEL:-/app/models/openfly-agent-7b}"
      OPENFLY_ATTN_IMPLEMENTATION: "\${OPENFLY_ATTN_IMPLEMENTATION:-eager}"
      HF_TOKEN: "\${HF_TOKEN:-}"
      HF_HOME: /app/.cache/huggingface
      TORCH_HOME: /app/.cache/torch
    command:
      - python3
      - neurons/miner.py
      - --netuid
      - "\${NETUID:-1}"
      - --subtensor.chain_endpoint
      - "\${SUBTENSOR_CHAIN_ENDPOINT:-ws://127.0.0.1:9944}"
      - --wallet.name
      - "\${MINER_WALLET_NAME:-miner}"
      - --wallet.hotkey
      - "\${MINER_WALLET_HOTKEY:-default}"
      - --axon.port
      - "\${MINER_AXON_PORT:-8091}"
      - --logging.debug
$external_axon_args
    ports:
      - "\${MINER_AXON_PORT:-8091}:\${MINER_AXON_PORT:-8091}"
    volumes:
      - ./logs:/app/logs:rw
      - ./wallets:/root/.bittensor/wallets:rw
      - ./OpenFly-Platform:/app/OpenFly-Platform:ro
      - ./models:/app/models:ro
      - ./logs/miner-hf-cache:/app/.cache/huggingface:rw
      - ./logs/miner-torch-cache:/app/.cache/torch:rw
EOF
}

build_image() {
  bold "Building miner image"
  require_docker
  dkr compose -f "$COMPOSE_FILE" build subnet-miner
}

rebuild_image() {
  bold "Rebuilding miner image"
  require_docker
  dkr compose -f "$COMPOSE_FILE" build --pull --no-cache subnet-miner
}

ensure_image() {
  dkr image inspect "$IMAGE_NAME" >/dev/null 2>&1 || build_image
}

wallet_file() {
  printf 'wallets/%s/hotkeys/%s' "$MINER_WALLET_NAME" "$MINER_WALLET_HOTKEY"
}

wallet_exists() {
  [[ -s "$(wallet_file)" ]]
}

require_wallet() {
  wallet_exists || die "Miner hotkey is missing: $(pwd)/$(wallet_file). Run install first."
}

run_wallet_cli() {
  dkr run --rm -it \
    --entrypoint btcli \
    -v "$PWD/wallets:/root/.bittensor/wallets:rw" \
    "$IMAGE_NAME" \
    wallet "$@" --wallet-path /root/.bittensor/wallets
}

delete_wallet() {
  local target="wallets/$MINER_WALLET_NAME"
  [[ -d "$target" ]] && rm -rf -- "$target"
  mkdir -p wallets
}

create_wallet() {
  delete_wallet
  bold "Creating coldkey"
  warn "Save the seed phrase shown by btcli."
  run_wallet_cli new_coldkey --wallet.name "$MINER_WALLET_NAME"

  bold "Creating hotkey"
  warn "Save the seed phrase shown by btcli."
  run_wallet_cli new_hotkey --wallet.name "$MINER_WALLET_NAME" --wallet.hotkey "$MINER_WALLET_HOTKEY"
}

restore_wallet() {
  local seed_file
  delete_wallet
  seed_file="$(mktemp .wallet-seed.XXXXXX)"
  chmod 600 "$seed_file"
  printf '%s' "$RESTORE_MNEMONIC" > "$seed_file"

  bold "Restoring coldkey and hotkey"
  if ! dkr run --rm \
    --entrypoint python3 \
    -e MINER_WALLET_NAME="$MINER_WALLET_NAME" \
    -e MINER_WALLET_HOTKEY="$MINER_WALLET_HOTKEY" \
    -v "$PWD/wallets:/root/.bittensor/wallets:rw" \
    -v "$PWD/$seed_file:/tmp/wallet-seed:ro" \
    "$IMAGE_NAME" \
    -c '
import contextlib
import os
from pathlib import Path
from bittensor_wallet import Wallet

wallet_path = Path("/root/.bittensor/wallets")
mnemonic = Path("/tmp/wallet-seed").read_text(encoding="utf-8").strip()
wallet = Wallet(
    name=os.environ["MINER_WALLET_NAME"],
    hotkey=os.environ["MINER_WALLET_HOTKEY"],
    path=str(wallet_path),
)
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        wallet.regenerate_coldkey(mnemonic=mnemonic, use_password=False, overwrite=True)
        wallet.regenerate_hotkey(mnemonic=mnemonic, use_password=False, overwrite=True)
print("Wallet restored.")
'
  then
    rm -f "$seed_file"
    return 1
  fi
  rm -f "$seed_file"
}

setup_wallet() {
  local choice
  printf '\nWallet\n'
  printf '  1) Create new coldkey + hotkey\n'
  printf '  2) Restore coldkey + hotkey from one seed phrase\n'
  printf 'Wallet action [2]: '
  read -r choice
  choice="${choice:-2}"

  case "$choice" in
    1) create_wallet ;;
    2)
      RESTORE_MNEMONIC="$(ask_secret "Seed phrase for coldkey and hotkey")"
      [[ -n "$RESTORE_MNEMONIC" ]] || die "Seed phrase is required."
      restore_wallet
      ;;
    *) die "Invalid wallet action." ;;
  esac

  wallet_exists || die "Miner hotkey was not created. Expected: $(wallet_file)"
  green "Wallet ready: $(wallet_file)"
}

wallet_address() {
  local file="$1"
  [[ -f "$file" ]] && sed -nE 's/.*"ss58Address"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$file" | head -n 1
}

print_wallet_status() {
  local cold hot
  cold="$(wallet_address "wallets/$MINER_WALLET_NAME/coldkeypub.txt")"
  hot="$(wallet_address "wallets/$MINER_WALLET_NAME/hotkeys/${MINER_WALLET_HOTKEY}pub.txt")"
  printf '\nWallet status\n'
  [[ -n "$cold" ]] && printf '  Coldkey: %s\n' "$cold"
  [[ -n "$hot" ]] && printf '  Hotkey:  %s\n' "$hot"
}

wallet_balance() {
  dkr_timeout "$DEFAULT_WALLET_BALANCE_TIMEOUT_SECONDS" run --rm \
    --entrypoint python3 \
    -e SUBTENSOR_CHAIN_ENDPOINT="$SUBTENSOR_CHAIN_ENDPOINT" \
    -e MINER_WALLET_NAME="$MINER_WALLET_NAME" \
    -v "$PWD/wallets:/root/.bittensor/wallets:rw" \
    "$IMAGE_NAME" \
    -c '
import os
import sys
import bittensor as bt

wallet = bt.wallet(name=os.environ["MINER_WALLET_NAME"], path="/root/.bittensor/wallets")
subtensor = bt.subtensor(network=os.environ["SUBTENSOR_CHAIN_ENDPOINT"])
balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
tao = float(getattr(balance, "tao", balance))
print(f"Free balance: {tao:.9f} TAO")
sys.exit(0 if tao > 0 else 2)
'
}

hotkey_registered() {
  dkr run --rm \
    -e NETUID="$NETUID" \
    -e SUBTENSOR_CHAIN_ENDPOINT="$SUBTENSOR_CHAIN_ENDPOINT" \
    -e MINER_WALLET_NAME="$MINER_WALLET_NAME" \
    -e MINER_WALLET_HOTKEY="$MINER_WALLET_HOTKEY" \
    -v "$PWD/wallets:/root/.bittensor/wallets:rw" \
    "$IMAGE_NAME" \
    python3 -c '
import os
import sys
import bittensor as bt

wallet = bt.wallet(
    name=os.environ["MINER_WALLET_NAME"],
    hotkey=os.environ["MINER_WALLET_HOTKEY"],
    path="/root/.bittensor/wallets",
)
subtensor = bt.subtensor(network=os.environ["SUBTENSOR_CHAIN_ENDPOINT"])
metagraph = subtensor.metagraph(int(os.environ["NETUID"]))
sys.exit(0 if wallet.hotkey.ss58_address in metagraph.hotkeys else 2)
' >/dev/null 2>&1
}

register_hotkey() {
  bold "Registering miner hotkey"
  dkr run --rm -it \
    --entrypoint btcli \
    -v "$PWD/wallets:/root/.bittensor/wallets:rw" \
    "$IMAGE_NAME" \
    subnet register \
      --wallet-name "$MINER_WALLET_NAME" \
      --wallet-path /root/.bittensor/wallets \
      --hotkey "$MINER_WALLET_HOTKEY" \
      --netuid "$NETUID" \
      --network "$SUBTENSOR_CHAIN_ENDPOINT" \
      --no-prompt -y
}

register_and_start() {
  print_wallet_status
  wallet_balance || warn "Could not confirm positive balance. Registration may fail if the coldkey has no funds."
  hotkey_registered && info "Hotkey is already registered on netuid $NETUID." || register_hotkey
  start_miner
}

collect_install_settings() {
  ensure_external_ip
  printf '\nMiner setup\n'
  printf '  netuid: %s\n' "$NETUID"
  printf '  RPC:    %s\n' "$SUBTENSOR_CHAIN_ENDPOINT"
  printf '  axon:   %s:%s\n' "${MINER_EXTERNAL_IP:-unknown}" "$MINER_EXTERNAL_PORT"
  MINER_WALLET_NAME="$(ask "Wallet coldkey name" "$MINER_WALLET_NAME")"
  MINER_WALLET_HOTKEY="$(ask "Wallet hotkey name" "$MINER_WALLET_HOTKEY")"
}

start_miner() {
  require_wallet
  require_docker
  ensure_external_ip
  set_env_value "MINER_EXTERNAL_IP" "$MINER_EXTERNAL_IP"
  set_env_value "MINER_EXTERNAL_PORT" "$MINER_EXTERNAL_PORT"
  write_compose
  ensure_image
  choose_backend
  write_compose
  hotkey_registered || die "Miner hotkey '$MINER_WALLET_NAME/$MINER_WALLET_HOTKEY' is not registered on netuid $NETUID."
  open_axon_port

  bold "Starting miner"
  dkr compose -f "$COMPOSE_FILE" up -d subnet-miner
  dkr compose -f "$COMPOSE_FILE" ps
  check_axon_port
  green "Miner is running. Use logs to watch it."
}

install_miner() {
  ensure_ubuntu_host
  install_base_packages
  install_docker
  prepare_repo
  load_env
  set_defaults
  collect_install_settings
  write_env
  write_compose
  open_axon_port
  build_image
  setup_wallet

  if ask_yes_no "Register hotkey and start miner now?" "y"; then
    register_and_start
  fi
}

start_existing_miner() {
  load_install
  start_miner
}

stop_miner() {
  load_install
  [[ -f "$COMPOSE_FILE" ]] || die "Compose file not found. Run start first."
  dkr compose -f "$COMPOSE_FILE" stop subnet-miner
}

show_logs() {
  load_install
  [[ -f "$COMPOSE_FILE" ]] || die "Compose file not found. Run start first."
  dkr compose -f "$COMPOSE_FILE" logs -f subnet-miner --tail 120
}

update_node() {
  load_install
  git pull --ff-only
  git submodule update --init --recursive
  write_compose
  rebuild_image
  start_miner
}

delete_node() {
  ensure_ubuntu_host
  if [[ -d "$DEFAULT_INSTALL_DIR" ]]; then
    cd "$DEFAULT_INSTALL_DIR"
    [[ -f "$COMPOSE_FILE" ]] && dkr compose -f "$COMPOSE_FILE" down --volumes --remove-orphans || true
  fi

  local container_ids
  container_ids="$(dkr ps -aq --filter "ancestor=$IMAGE_NAME" 2>/dev/null || true)"
  [[ -n "$container_ids" ]] && dkr rm -f $container_ids >/dev/null 2>&1 || true
  dkr image rm -f "$IMAGE_NAME" >/dev/null 2>&1 || true

  if ask_yes_no "Delete install directory $DEFAULT_INSTALL_DIR including wallets and logs?" "n"; then
    cd "$(dirname "$DEFAULT_INSTALL_DIR")"
    rm -rf -- "$DEFAULT_INSTALL_DIR"
    green "Node directory, containers, and image deleted."
  else
    green "Containers and image deleted. Files kept at $DEFAULT_INSTALL_DIR"
  fi
}

node_status() {
  local dc cid state
  [[ -f "$DEFAULT_INSTALL_DIR/$COMPOSE_FILE" ]] || { printf 'not installed'; return; }
  dc="$(docker_cmd)" || { printf 'docker offline'; return; }
  cd "$DEFAULT_INSTALL_DIR"
  cid="$($dc compose -f "$COMPOSE_FILE" ps -a -q subnet-miner 2>/dev/null || true)"
  [[ -n "$cid" ]] || { printf 'offline'; return; }
  state="$($dc inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || true)"
  printf '%s' "${state:-offline}"
}

main_menu() {
  local old_pwd choice
  ensure_ubuntu_host
  while true; do
    old_pwd="$(pwd)"
    if [[ -f "$DEFAULT_INSTALL_DIR/.env" ]]; then
      cd "$DEFAULT_INSTALL_DIR"
      load_env
      cd "$old_pwd"
    fi
    set_defaults
    printf '\n\033[1;36mKONNEX MINER\033[0m\n'
    printf '  status: %s\n' "$(node_status)"
    printf '  dir:    %s\n' "$DEFAULT_INSTALL_DIR"
    printf '  netuid: %s\n' "$NETUID"
    printf '\n'
    printf '  1) install\n'
    printf '  2) start\n'
    printf '  3) stop\n'
    printf '  4) logs\n'
    printf '  5) update\n'
    printf '  6) delete\n'
    printf '  0) exit\n'
    printf '> '
    read -r choice
    case "$choice" in
      1) install_miner ;;
      2) start_existing_miner ;;
      3) stop_miner ;;
      4) show_logs ;;
      5) update_node ;;
      6) delete_node ;;
      0) return ;;
      *) warn "Choose 1, 2, 3, 4, 5, 6, or 0." ;;
    esac
  done
}

print_usage() {
  cat <<EOF
Usage: ./konnex.miner.sh [command]

Commands:
  install   install miner and set up wallet
  start     start existing miner
  stop      stop miner
  logs      show miner logs
  update    pull repo, rebuild image, restart miner
  delete    delete containers/image and optionally node directory
  menu      show menu (default)
EOF
}

main() {
  case "${1:-menu}" in
    install) install_miner ;;
    start) start_existing_miner ;;
    stop) stop_miner ;;
    logs) show_logs ;;
    update) update_node ;;
    delete) delete_node ;;
    menu) main_menu ;;
    help|--help|-h) print_usage ;;
    *) print_usage; die "Unknown command: $1" ;;
  esac
}

main "$@"
