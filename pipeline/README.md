# TextureLab Batch (v3.0.0)

Cálculo em lote, sem interface, dos descritores de textura de pavimento a partir de
varreduras 3dT (LAZ). Núcleo conferido contra as normas ISO 13473-1:2019, 13473-4:2024,
25178-2/-3 e 13565-2.

```bash
pip install -r requirements.txt
python texturelab_batch.py --input /caminho/dos/laz --output /caminho/resultados
```

Opções: `--pattern "*B6*MP1*.laz"` (filtro), `--workers N` (paralelismo), `--skip-done` (retomar).

Documentação completa:
- [White paper (PT)](../docs/WHITE_PAPER_PT.md) · [White paper (EN)](../docs/WHITE_PAPER_EN.md)
- [Diagnóstico e correções](../docs/DIAGNOSTICO.md)
