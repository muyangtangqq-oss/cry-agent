#!/bin/bash
# One-shot macOS (Apple Silicon) deployment for cry-agent.
#
# Installs system + Python dependencies, clones or updates the repo, copies
# trained models from another Mac, and installs a launchd service that keeps
# Tachyone + the API server running.
#
# Typical use on the Mac mini:
#   ./deploy_mac.sh --models-from kl@macbook.local
#
# Optional data copy for replay/retraining (about 685 MB):
#   ./deploy_mac.sh --models-from kl@macbook.local \
#                   --data-from kl@macbook.local:/Users/kl/cry-agent/data
#
# Run ./deploy_mac.sh --help for all options.

set -euo pipefail

REPO_URL="https://github.com/muyangtangqq-oss/cry-agent.git"
BRANCH="main"
APP_DIR="$HOME/cry-agent"
PORT="8000"
TACHYONE_PORT="8901"
LABEL="com.cryagent.api"
MODELS_FROM=""
DATA_FROM=""
JUDGE_PROVIDER="${CRY_AGENT_JUDGE_PROVIDER:-deepseek}"
WITH_LIVE_DEPS=1
INSTALL_LAUNCHD=1
ALLOW_MISSING_MODELS=0
SKIP_DEPS=0

CONF_DIR="$HOME/.config/cry-agent"
MODEL_DIR="$CONF_DIR/models"
ENV_FILE="$CONF_DIR/env"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/cry-agent"
RUNNER="$APP_DIR/run_service.sh"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./deploy_mac.sh [options]

  --models-from SRC     Copy trained models from SRC. SRC is either
                        "user@host" (reads ~/.config/cry-agent/models/) or
                        "user@host:/absolute/path/to/models" or a local dir.
  --data-from SRC       Optional: copy the 685 MB data/ tree for replay or
                        retraining. Needs an explicit remote path, e.g.
                        "user@host:/Users/kl/cry-agent/data".
  --judge-provider NAME deepseek (default), qwen, glm, kimi, ollama, openai.
  --app-dir DIR         Install directory (default: ~/cry-agent).
  --port N              API port (default: 8000).
  --tachyone-port N     Local judge port (default: 8901).
  --repo URL            Git remote (default: $REPO_URL).
  --branch NAME         Git branch (default: main).
  --no-live-deps        Skip sounddevice/opencv/mediapipe.
  --skip-deps           Skip Homebrew and Python dependency installation
                        (useful when the environment is already provisioned).
  --no-launchd          Do not install the launchd service.
  --allow-missing-models
                        Continue even if no classifier is present.
  -h, --help            Show this help.

The API key is taken from CRY_AGENT_JUDGE_API_KEY if set, otherwise from
~/.config/magpie/providers.json (copied separately if you use that).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-from) MODELS_FROM="${2:-}"; shift 2 ;;
    --data-from) DATA_FROM="${2:-}"; shift 2 ;;
    --judge-provider) JUDGE_PROVIDER="${2:-}"; shift 2 ;;
    --app-dir) APP_DIR="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --tachyone-port) TACHYONE_PORT="${2:-}"; shift 2 ;;
    --repo) REPO_URL="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    --no-live-deps) WITH_LIVE_DEPS=0; shift ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --no-launchd) INSTALL_LAUNCHD=0; shift ;;
    --allow-missing-models) ALLOW_MISSING_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "this script targets macOS"
[[ -n "$APP_DIR" ]] || die "--app-dir must not be empty"
case "$JUDGE_PROVIDER" in
  jev|deepseek|qwen|glm|kimi|ollama|openai) ;;
  *) die "unknown --judge-provider '$JUDGE_PROVIDER'" ;;
esac
RUNNER="$APP_DIR/run_service.sh"

# ---------------------------------------------------------------- prerequisites
command -v git >/dev/null 2>&1 || die "git is missing; install the Xcode command line tools"

if [[ "$SKIP_DEPS" -eq 0 ]]; then
  log "Checking Xcode command line tools"
  if ! xcode-select -p >/dev/null 2>&1; then
    die "run 'xcode-select --install', finish the dialog, then rerun this script"
  fi

  log "Checking Homebrew"
  if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew is required. Install it, then rerun:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
  fi

  for pkg in ffmpeg libsndfile portaudio; do
    if brew list --versions "$pkg" >/dev/null 2>&1; then
      log "brew: $pkg already installed"
    else
      log "brew: installing $pkg"
      brew install "$pkg"
    fi
  done

  if command -v uv >/dev/null 2>&1; then
    log "uv: $(uv --version)"
  else
    log "brew: installing uv"
    brew install uv
  fi
else
  warn "skipping dependency installation (--skip-deps)"
fi

# ------------------------------------------------------------------- source tree
if [[ -d "$APP_DIR/.git" ]]; then
  log "Updating source in $APP_DIR"
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
elif [[ -e "$APP_DIR" && -n "$(ls -A "$APP_DIR" 2>/dev/null || true)" ]]; then
  die "$APP_DIR exists but is not a git checkout; move it away or use --app-dir"
else
  log "Cloning $REPO_URL -> $APP_DIR"
  mkdir -p "$(dirname "$APP_DIR")"
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# ---------------------------------------------------------------- python packages
if [[ "$SKIP_DEPS" -eq 0 ]]; then
  log "Installing Python dependencies (this can take a few minutes)"
  if ! uv sync --no-install-project; then
    warn "uv sync --no-install-project failed; retrying plain uv sync"
    uv sync
  fi

  if [[ "$WITH_LIVE_DEPS" -eq 1 ]]; then
    log "Installing camera/microphone extras"
    if ! uv pip install sounddevice opencv-python mediapipe; then
      warn "camera/microphone extras failed to install; replay/API modes still work"
    fi
  fi
fi

# ------------------------------------------------------------------------ models
backup_dir() {
  local src="$1"
  [[ -d "$src" ]] || return 0
  local dst="${src}.bak.$(date +%Y%m%d-%H%M%S)"
  log "Backing up $src -> $dst"
  cp -a "$src" "$dst"
}

copy_tree() {
  local src="$1" dst="$2" default_remote="$3"
  mkdir -p "$dst"
  if [[ "$src" == /* ]]; then
    [[ -d "$src" ]] || die "local source directory not found: $src"
    log "Copying $src -> $dst"
    rsync -a --progress "$src"/ "$dst"/
  elif [[ "$src" == *:* ]]; then
    log "Copying $src/ -> $dst"
    rsync -a --progress "$src"/ "$dst"/
  else
    log "Copying $src:$default_remote/ -> $dst"
    rsync -a --progress "$src:$default_remote"/ "$dst"/
  fi
}

if [[ -n "$MODELS_FROM" ]]; then
  backup_dir "$MODEL_DIR"
  copy_tree "$MODELS_FROM" "$MODEL_DIR" '~/.config/cry-agent/models'
else
  log "No --models-from given; using existing models in $MODEL_DIR"
fi

if [[ ! -f "$MODEL_DIR/cry_classifier.joblib" ]]; then
  if [[ "$ALLOW_MISSING_MODELS" -eq 1 ]]; then
    warn "no classifier in $MODEL_DIR; the API will fail until you add one"
  else
    die "no classifier in $MODEL_DIR.
Copy it from the training Mac:
  ./deploy_mac.sh --models-from user@host
or pass --allow-missing-models if you plan to train here."
  fi
fi

if [[ -n "$DATA_FROM" ]]; then
  backup_dir "$APP_DIR/data"
  copy_tree "$DATA_FROM" "$APP_DIR/data" '~/cry-agent/data'
fi

# ------------------------------------------------------------------ env + runner
mkdir -p "$CONF_DIR" "$LOG_DIR" "$HOME/Library/LaunchAgents"
if [[ -f "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%Y%m%d-%H%M%S)"
fi
OLD_JUDGE_KEY=""
if [[ -f "$ENV_FILE" ]]; then
  OLD_JUDGE_KEY="$(grep '^CRY_AGENT_JUDGE_API_KEY=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
fi
JUDGE_KEY="${CRY_AGENT_JUDGE_API_KEY:-$OLD_JUDGE_KEY}"
{
  echo "# generated by deploy_mac.sh"
  echo "CRY_AGENT_JUDGE_PROVIDER=$JUDGE_PROVIDER"
  if [[ -n "$JUDGE_KEY" ]]; then
    echo "CRY_AGENT_JUDGE_API_KEY=$JUDGE_KEY"
  fi
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

log "Writing $RUNNER"
cat > "$RUNNER" <<EOF
#!/bin/zsh
set -euo pipefail
cd "$APP_DIR"
mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

cleanup() {
  for pid in "\${UVICORN_PID:-}" "\${TACHYONE_PID:-}"; do
    [[ -n "\$pid" ]] && kill "\$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

"$APP_DIR/.venv/bin/python" "$APP_DIR/tachyone_server.py" "$TACHYONE_PORT" \\
  >> "$LOG_DIR/tachyone.log" 2>&1 &
TACHYONE_PID=\$!

for _ in {1..30}; do
  if nc -z 127.0.0.1 "$TACHYONE_PORT" 2>/dev/null; then
    break
  fi
  sleep 1
done

"$APP_DIR/.venv/bin/python" -m uvicorn api_server:app \\
  --host 0.0.0.0 --port "$PORT" >> "$LOG_DIR/api.log" 2>&1 &
UVICORN_PID=\$!
wait "\$UVICORN_PID"
EOF
chmod +x "$RUNNER"

# -------------------------------------------------------------------- launchd
if [[ "$INSTALL_LAUNCHD" -eq 0 ]]; then
  log "Skipping launchd (--no-launchd)"
  echo
  echo "Start it in the foreground with:"
  echo "  $RUNNER"
  exit 0
fi

log "Writing $PLIST"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>exec "$RUNNER"</string>
  </array>
  <key>WorkingDirectory</key><string>$APP_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOG_DIR/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/launchd.err.log</string>
</dict>
</plist>
EOF

log "Reloading launchd service $LABEL"
launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
if ! launchctl bootstrap "gui/$(id -u)" "$PLIST" >/dev/null 2>&1; then
  warn "launchctl bootstrap failed; falling back to launchctl load"
  launchctl load -w "$PLIST"
fi
launchctl kickstart -k "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true

log "Waiting for the API on port $PORT"
ok=0
for _ in {1..45}; do
  if curl -fsS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done

echo
if [[ "$ok" -eq 1 ]]; then
  log "Deployment complete"
  LOCAL_HOST="$(scutil --get LocalHostName 2>/dev/null || hostname)"
  echo "  Dashboard: http://$LOCAL_HOST.local:$PORT/dashboard"
  echo "  Local:     http://127.0.0.1:$PORT/dashboard"
else
  warn "service installed but the API did not answer in 45 s"
  echo "Check logs:"
  echo "  tail -50 $LOG_DIR/api.log"
  echo "  tail -50 $LOG_DIR/tachyone.log"
  echo "  tail -50 $LOG_DIR/launchd.err.log"
  exit 1
fi

cat <<EOF

Service control:
  launchctl kickstart -k gui/$(id -u)/$LABEL   # restart
  launchctl bootout   gui/$(id -u) $PLIST      # stop

First camera/microphone run must happen in a Terminal on the mini's desktop
so macOS can show the privacy prompts:
  $APP_DIR/.venv/bin/python $APP_DIR/live_camera.py --camera 0 --mic-device 0
EOF
