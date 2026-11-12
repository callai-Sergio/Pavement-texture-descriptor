# Roadmap do TextureLab

Estado em 2026-11-12: core 3.5.0, `Resultados_v3.5` é o resultado atual. Todos os passos do resumo de 2026-11-10 estão concluídos.

## Em aberto

- [ ] Exportar a tabela SRT (pêndulo) junto com as superfícies, para a aba Pêndulo do simulador (precisa ser anonimizada; ISO 4 não tem dado de pêndulo).
- [ ] Se o simulador precisar de mais perfis ou cadeias para a aba Filtros, ampliar `tools/exportar_superficies.py` (formato em `docs/FORMATO_SUPERFICIES.md`).
- [ ] Visualizador no servidor: não há mais serviço systemd. Decidir se volta a ter um serviço ou se fica só no PC.

## Convenções

- Uma branch e um PR por mudança; versão em `pyproject.toml` e linha no `CHANGELOG.md`.
- Mudou o core → recalcular com `servidor.sh iniciar dados/LAZ dados/Resultados_vX.Y` e manter só as duas últimas versões de resultados.
