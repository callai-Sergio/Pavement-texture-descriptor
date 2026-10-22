# baixar_resultados.ps1 - Copia para o PC os projetos (.tlproj) calculados no servidor.
#
# Depois disso o app abre os projetos localmente, sem acesso ao servidor (ver abrir_app.ps1).
# Só baixa o que é novo ou mudou (compara tamanho e data). Usa o ssh/scp que já vem no Windows.
#
# Uso (PowerShell, na pasta do app):   .\tools\baixar_resultados.ps1
#   ou clique duas vezes em tools\baixar_resultados.bat
# Opções: -Destino "D:\outra\pasta"  -Servidor "usuario@host"  -PastaRemota "/caminho/no/servidor"
param(
    [string]$Servidor = "sergio@137.226.169.235",
    [string]$PastaRemota = "/data/callai/workspace/tyron",
    [string]$Destino = "H:\TextureLab\projetos"
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "O comando ssh não foi encontrado. Instale o 'Cliente OpenSSH' do Windows:" -ForegroundColor Red
    Write-Host "  Configurações > Aplicativos > Recursos opcionais > Adicionar > Cliente OpenSSH"
    exit 1
}
New-Item -ItemType Directory -Force -Path $Destino | Out-Null

Write-Host "Consultando $Servidor ..."
$remoto = 'cd {0} && for f in Resultados*.tlproj; do [ -f $f ] && stat -c %n:%s:%Y $f; done; echo LOTE=$(systemctl --user is-active texturelab-lote 2>/dev/null)' -f $PastaRemota
$linhas = @(ssh $Servidor $remoto)
if ($LASTEXITCODE -ne 0) { Write-Host "Falha ao conectar ao servidor (rede/VPN ou senha)." -ForegroundColor Red; exit 1 }

$lote = ($linhas | Where-Object { $_ -like "LOTE=*" }) -replace "LOTE=", ""
if ($lote -eq "active") {
    Write-Host "Atenção: há um lote rodando no servidor; o pacote dele só aparece aqui quando terminar." -ForegroundColor Yellow
}
$pacotes = $linhas | Where-Object { $_ -match "^(.+\.tlproj):(\d+):(\d+)$" }
if (-not $pacotes) { Write-Host "Nenhum pacote .tlproj em $PastaRemota."; exit 0 }

$novos = 0
foreach ($p in $pacotes) {
    $null = $p -match "^(.+\.tlproj):(\d+):(\d+)$"
    $nome, $tam, $epoch = $Matches[1], [int64]$Matches[2], [int64]$Matches[3]
    $data = [DateTimeOffset]::FromUnixTimeSeconds($epoch).UtcDateTime
    $local = Join-Path $Destino $nome
    if ((Test-Path $local) -and ((Get-Item $local).Length -eq $tam) -and
        ([math]::Abs(((Get-Item $local).LastWriteTimeUtc - $data).TotalSeconds) -lt 2)) {
        Write-Host ("  {0,-32} já está atualizado" -f $nome)
        continue
    }
    Write-Host ("  {0,-32} baixando {1:N0} MB ..." -f $nome, ($tam / 1MB)) -ForegroundColor Cyan
    $tmp = "$local.part"
    scp "${Servidor}:$PastaRemota/$nome" "$tmp"
    if ($LASTEXITCODE -ne 0 -or (Get-Item $tmp).Length -ne $tam) {
        Write-Host "  falhou: $nome (tente de novo)" -ForegroundColor Red
        Remove-Item -ErrorAction SilentlyContinue $tmp
        continue
    }
    Move-Item -Force $tmp $local
    (Get-Item $local).LastWriteTimeUtc = $data
    $novos++
}
Write-Host ""
Write-Host "Pronto: $novos pacote(s) novo(s) em $Destino" -ForegroundColor Green
Write-Host "Para abrir: tools\abrir_app.bat (funciona sem o servidor)."
