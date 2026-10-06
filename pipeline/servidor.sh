#!/usr/bin/env bash
# servidor.sh – Roda o lote do TextureLab no servidor compartilhado com uso de recursos controlado.
#
#   ./servidor.sh instalar                    cria o venv próprio (pipeline/.venv) e grava as versões
#   ./servidor.sh iniciar [ENTRADA] [SAIDA]   começa o lote com limites de CPU, RAM e disco
#   ./servidor.sh status                      andamento, CPU e RAM em uso, estimativa de término
#   ./servidor.sh pausar | continuar          congela / retoma o cálculo sem perder nada
#   ./servidor.sh parar                       interrompe (com --skip-done o que terminou não é refeito)
#   ./servidor.sh log                         acompanha o log (Ctrl+C sai, o cálculo continua)
#
# Limites: edite abaixo ou passe por variável de ambiente, ex.: CPU_NUCLEOS=8 ./servidor.sh iniciar
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRABALHO="${TRABALHO:-/data/callai/workspace/tyron}"
ENTRADA_PADRAO="${ENTRADA_PADRAO:-$TRABALHO/LAZ}"
SAIDA_PADRAO="${SAIDA_PADRAO:-$TRABALHO/Resultados_v3}"

CPU_NUCLEOS="${CPU_NUCLEOS:-4}"        # teto de CPU (núcleos inteiros) e número de processos
RAM_MAX="${RAM_MAX:-25G}"              # acima disso o sistema encerra só este cálculo
RAM_LIVRE_MIN_GB="${RAM_LIVRE_MIN_GB:-30}"   # não inicia se o servidor tiver menos RAM disponível
DISCO_LIVRE_MIN_GB="${DISCO_LIVRE_MIN_GB:-20}"
EXTRA="${EXTRA:-}"                     # opções extras do pipeline, ex.: EXTRA="--config ajustes.json"

UNIDADE="texturelab-lote"
VENV="$AQUI/.venv"
PY="$VENV/bin/python"
ESTADO="$HOME/.texturelab_lote"        # guarda a saída e o log da última execução

die() { echo "ERRO: $*" >&2; exit 1; }
ativo() { systemctl --user is-active --quiet "$UNIDADE.service"; }

instalar() {
    python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -r "$AQUI/requirements.txt"
    "$PY" -m pip freeze > "$AQUI/.venv-versoes.txt"
    echo "venv pronto: $VENV"
    cat "$AQUI/.venv-versoes.txt"
}

iniciar() {
    local entrada="${1:-$ENTRADA_PADRAO}" saida="${2:-$SAIDA_PADRAO}"
    [ -x "$PY" ] || die "venv não encontrado: rode './servidor.sh instalar' primeiro"
    ativo && die "já existe um lote rodando ('./servidor.sh status')"
    [ -d "$entrada" ] || die "pasta de entrada não existe: $entrada"
    local livre disco
    livre=$(awk '/MemAvailable/ {printf "%d", $2/1e6}' /proc/meminfo)
    [ "$livre" -ge "$RAM_LIVRE_MIN_GB" ] || die "só ${livre} GB de RAM disponível (mínimo ${RAM_LIVRE_MIN_GB} GB). Tente mais tarde."
    mkdir -p "$saida"
    disco=$(df -BG --output=avail "$saida" | tail -1 | tr -dc 0-9)
    [ "$disco" -ge "$DISCO_LIVRE_MIN_GB" ] || die "só ${disco} GB livres em disco (mínimo ${DISCO_LIVRE_MIN_GB} GB)"
    local log="$saida/processamento_$(date +%Y%m%d_%H%M%S).log"
    systemctl --user reset-failed "$UNIDADE.service" 2>/dev/null || true
    # shellcheck disable=SC2086
    systemd-run --user --unit="$UNIDADE" --collect --quiet \
        -p CPUQuota="$((CPU_NUCLEOS * 100))%" -p MemoryMax="$RAM_MAX" -p MemorySwapMax=0 \
        -p Nice=19 -p IOSchedulingClass=idle \
        -p StandardOutput="append:$log" -p StandardError="append:$log" \
        -p WorkingDirectory="$saida" \
        "$PY" -u "$AQUI/texturelab_batch.py" --input "$entrada" --output "$saida" \
        --workers "$CPU_NUCLEOS" --skip-done --zip $EXTRA
    printf 'SAIDA=%q\nLOG=%q\nINICIO=%q\n' "$saida" "$log" "$(date +%s)" > "$ESTADO"
    echo "Lote iniciado: $CPU_NUCLEOS núcleos, RAM máx. $RAM_MAX, prioridade mínima de CPU e disco."
    echo "Entrada: $entrada"
    echo "Saída:   $saida"
    echo "Log:     $log"
    echo "Acompanhe com './servidor.sh status' ou './servidor.sh log'."
}

status() {
    [ -f "$ESTADO" ] || die "nenhum lote iniciado por este script"
    # shellcheck disable=SC1090
    source "$ESTADO"
    local estado
    if ! ativo; then
        estado="encerrado"
    elif [ "$(systemctl --user show "$UNIDADE.service" -p FreezerState --value)" = "frozen" ]; then
        estado="pausado"
    else
        estado="rodando"
    fi
    echo "Estado:  $estado   (unidade $UNIDADE)"
    echo "Saída:   $SAIDA"
    if ativo; then
        local mem cpu
        mem=$(systemctl --user show "$UNIDADE.service" -p MemoryCurrent --value)
        cpu=$(systemctl --user show "$UNIDADE.service" -p CPUUsageNSec --value)
        awk -v m="$mem" -v c="$cpu" -v t="$(( $(date +%s) - INICIO ))" 'BEGIN {
            printf "RAM:     %.1f GB agora\n", m/1e9;
            if (t > 0) printf "CPU:     %.1f núcleos em média desde o início\n", c/1e9/t }'
    fi
    local linha
    linha=$(grep -E '^\[[0-9]+/[0-9]+\]' "$LOG" | tail -1 || true)
    head -2 "$LOG" | grep -E 'arquivo\(s\)|skip-done' || true
    if [ -n "$linha" ]; then
        local feito total dec
        feito=$(sed -E 's/^\[([0-9]+)\/.*/\1/' <<< "$linha")
        total=$(sed -E 's/^\[[0-9]+\/([0-9]+)\].*/\1/' <<< "$linha")
        dec=$(( $(date +%s) - INICIO ))
        echo "Feitos:  $feito/$total   ($(( dec / 60 )) min decorridos)"
        if ativo && [ "$feito" -lt "$total" ]; then
            echo "Faltam:  ~$(( dec * (total - feito) / feito / 60 )) min (estimativa pelo ritmo atual)"
        fi
        echo "Último:  $linha"
    else
        echo "Feitos:  0 (primeiros arquivos em cálculo)"
    fi
    grep -E ': erro ' "$LOG" | sed 's/^/Erro:    /' || true
    grep -E '^Concluído|^Pacote|^Nada a recalcular' "$LOG" || true
}

case "${1:-}" in
    instalar)  instalar ;;
    iniciar)   shift; iniciar "$@" ;;
    status)    status ;;
    pausar)    systemctl --user freeze "$UNIDADE.service" && echo "Pausado (CPU liberada; RAM continua reservada)." ;;
    continuar) systemctl --user thaw "$UNIDADE.service" && echo "Retomado." ;;
    parar)     systemctl --user stop "$UNIDADE.service" && echo "Parado. Para retomar: './servidor.sh iniciar' (pula o que já terminou)." ;;
    log)       source "$ESTADO"; tail -f "$LOG" ;;
    *)         sed -n '2,11p' "$0"; exit 1 ;;
esac
