# Ver os resultados no PC, sem o servidor

O servidor calcula (`pipeline/servidor.sh`) e grava um pacote `.tlproj` por lote. O PC copia esses
pacotes para uma pasta local, e o app abre a partir dali, sem túnel nem servidor, inclusive offline.

## Instalação (uma vez)

No **PowerShell do PC**:

```powershell
# código do app (sem precisar de git); o mesmo bloco serve para atualizar depois
$zip = "$env:TEMP\texturelab.zip"; $src = "$env:TEMP\texturelab_src"
Invoke-WebRequest https://github.com/callai-Sergio/Pavement-texture-descriptor/archive/refs/heads/master.zip -OutFile $zip
Remove-Item $src -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip -DestinationPath $src
New-Item -ItemType Directory -Force H:\TextureLab\app | Out-Null
Copy-Item "$src\Pavement-texture-descriptor-master\*" H:\TextureLab\app -Recurse -Force
```

Com git: `git clone https://github.com/callai-Sergio/Pavement-texture-descriptor.git H:\TextureLab\app`.

Precisa de **Python 3.10 ou mais novo**. Se não tiver: `winget install Python.Python.3.12`.

## Uso

| Ação | Como |
|---|---|
| Copiar do servidor os projetos novos | clique duas vezes em `H:\TextureLab\app\tools\baixar_resultados.bat` |
| Abrir o app (local) | clique duas vezes em `H:\TextureLab\app\tools\abrir_app.bat` → abre http://localhost:8502 |

- `baixar_resultados` só baixa os pacotes **novos ou alterados** (compara tamanho e data). Precisa de
  rede/VPN até o servidor só nesse momento. Um lote ainda em andamento não aparece até terminar.
- `abrir_app` na primeira vez cria o ambiente Python (`TextureLab\.venv`) e instala só o visualizador.
  Depois abre direto. No app, a lista **"Projetos neste PC"** mostra os pacotes da pasta local.
- A porta local é 8502, para não colidir com o túnel do app no servidor (8501).

Outra pasta: `baixar_resultados.bat -Destino D:\Dados\TextureLab` e `abrir_app.bat -Projetos D:\Dados\TextureLab`.

Atualizar o app: repita o bloco de instalação (ou `git pull` em `H:\TextureLab\app`).
O ambiente Python é reaproveitado e só reinstala bibliotecas se `requirements-viewer.txt` mudar.
