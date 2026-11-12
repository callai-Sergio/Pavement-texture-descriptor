# Formato `texturelab-superficies` (versão 1)

Pacote JSON gerado por `tools/exportar_superficies.py` a partir de uma pasta de resultados do TextureLab.
É o **único contato** entre o TextureLab e o simulador visual: o simulador lê só este arquivo e não conhece
o formato interno dos resultados.

```bash
python tools/exportar_superficies.py dados/Resultados_v3.5            # → dados/Resultados_v3.5_superficies.json
python tools/exportar_superficies.py dados/Resultados_v3.5 -o superficies.json --comprimento-mm 100 --reducao 2
```

Roda no venv do pipeline. Só lê o projeto, não calcula descritores. Para 16 superfícies: ≈ 5,3 MB.

## Escolha da medição

De cada superfície (trecho + revestimento) entra **uma** medição: o ponto de medição (MP) cujo MPD é o mais
próximo da mediana da superfície; em empate, o menor MP. Só entram arquivos com `status = ok` e `visualizacao`.

## Estrutura

```
{
  "formato": "texturelab-superficies",
  "versao_formato": 1,
  "projeto": "Resultados_v3.5",        // nome da pasta de origem
  "versao_nucleo": "3.5.0",
  "cadeia": "SL5",
  "superficies": [ { ... }, ... ]       // ordenadas por (trecho, revestimento)
}
```

Cada item de `superficies`:

| Campo | Conteúdo |
|---|---|
| `arquivo`, `trecho`, `revestimento`, `mp`, `data` | identificação da medição escolhida |
| `nx`, `ny`, `passo_mm`, `escala_um`, `z_b64` | superfície SL5: matriz `ny × nx` (largura × via), alturas em int16 little-endian, em base64, `escala_um` µm por unidade (1,0), média zero |
| `frac_invalidos` | fração de pontos sem dado antes do preenchimento (o buraco vira o ponto mais baixo, não um pico) |
| `parametros` | MPD, g_factor_media, perfil_Rsk_media, SL5_Sq/Ssk/Sku/Sk/Spk/Svk, SF_Sq, H_macro da medição **inteira**; `null` se ausente |
| `perfis` | ver abaixo; chaves ausentes nos resultados antigos são omitidas |

O recorte é central no sentido da via (`--comprimento-mm`, padrão 100 mm) e a superfície é reduzida pela média
de blocos `N × N` (`--reducao`, padrão 2). Com os padrões: passo 0,176 mm, ≈ 190 × 568 pontos.

### `perfis`

- `nativo`: perfil original de 100 mm. `passo_mm` (0,011), `y_mm` (posição transversal), `inicio_mm`, mais um
  bloco codificado (abaixo).
- `cadeia_A`: `passo_mm` (0,5), `faixas`, `y_mm` (5 posições) e dois blocos codificados de forma `[5, n]`:
  `limpo` (bruto limpo) e `passa_baixa` (após o Butterworth 2,40 mm).
- `abbott`: por cadeia `MICRO`, `SF`, `SL5`: `mr_pct`, `altura_mm` (201 pontos), `hist_bordas_mm` (121) e
  `hist_contagem` (120).

**Bloco codificado** (perfis): `{ "forma": [...], "escala_um": 1.0, "sem_dado": -32768, "b64": "..." }`. Valores int16
little-endian em µm; o valor `sem_dado` marca ponto sem medida (NaN). Decodificar: `valor_mm = int16 × escala_um / 1000`.

## Leitura em Python

```python
import json, exportar_superficies as ex
d = json.load(open("superficies.json"))
s = d["superficies"][0]
z = ex.decodificar(s)                                   # (ny, nx), mm
perfil = ex.decodificar_perfil(s["perfis"]["nativo"])    # (n,), mm, NaN onde não há dado
```

Em JavaScript: `Int16Array` sobre o resultado de `atob`/`Uint8Array` (little-endian), multiplicando por `escala_um / 1000`.

## Compatibilidade

Mudanças que quebram a leitura aumentam `versao_formato`. Campos novos opcionais não aumentam.
