# abrir_app.ps1 - Abre o visualizador TextureLab no PC, sem servidor e sem túnel.
#
# Na primeira vez cria um ambiente Python próprio (TextureLab\.venv) e instala só o necessário para
# visualizar. Depois abre o app no navegador (http://localhost:8502) com a lista de projetos da pasta local.
#
# Uso: clique duas vezes em tools\abrir_app.bat   ou   .\tools\abrir_app.ps1 [-Projetos "D:\pasta"] [-Porta 8502]
param(
    [string]$Projetos = "H:\TextureLab\projetos",
    [int]$Porta = 8502            # 8502 para não colidir com o túnel do servidor (8501)
)
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
$app = Join-Path $raiz "TextureLab\app.py"
$req = Join-Path $raiz "TextureLab\requirements-viewer.txt"
$venv = Join-Path $raiz "TextureLab\.venv"
$py = Join-Path $venv "Scripts\python.exe"

function Find-Python {
    foreach ($c in @(@("py", "-3"), @("python"), @("python3"))) {
        $exe = Get-Command $c[0] -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        try { $v = & $c[0] @($c | Select-Object -Skip 1) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null } catch { continue }
        if ($LASTEXITCODE -eq 0 -and $v -match "^3\.(\d+)$" -and [int]$Matches[1] -ge 10) { return ,$c }
    }
    return $null
}

if (-not (Test-Path $py)) {
    $base = Find-Python
    if (-not $base) {
        Write-Host "Python 3.10 ou mais novo não encontrado." -ForegroundColor Red
        Write-Host "Instale com:  winget install Python.Python.3.12   (ou em python.org, marcando 'Add to PATH')"
        Write-Host "Depois rode este script de novo."
        exit 1
    }
    Write-Host "Criando o ambiente do app (só na primeira vez) ..." -ForegroundColor Cyan
    & $base[0] @($base | Select-Object -Skip 1) -m venv $venv
}
# (re)instala quando requirements-viewer.txt muda
$hash = (Get-FileHash $req).Hash
$marca = Join-Path $venv "requirements.hash"
if (-not (Test-Path $marca) -or (Get-Content $marca) -ne $hash) {
    Write-Host "Instalando bibliotecas do visualizador ..." -ForegroundColor Cyan
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r $req --quiet
    if ($LASTEXITCODE -ne 0) { Write-Host "Falha na instalação." -ForegroundColor Red; exit 1 }
    Set-Content -Path $marca -Value $hash
}

New-Item -ItemType Directory -Force -Path $Projetos | Out-Null
$env:TEXTURELAB_PROJETOS = $Projetos
Write-Host "Abrindo o TextureLab em http://localhost:$Porta  (projetos em $Projetos)" -ForegroundColor Green
Write-Host "Para fechar, feche esta janela ou aperte Ctrl+C."
& $py -m streamlit run $app --server.address 127.0.0.1 --server.port $Porta --browser.gatherUsageStats false
