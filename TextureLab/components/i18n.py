"""
i18n.py – Textos do visualizador em inglês, alemão e português.

t("chave", **valores) devolve o texto no idioma atual (set_language). Chave ausente em um idioma
cai no inglês; chave desconhecida devolve a própria chave (fácil de achar na tela).
"""
from __future__ import annotations

LANGUAGES = {"en": "English", "de": "Deutsch", "pt": "Português"}
DEFAULT_LANGUAGE = "en"
_lang = DEFAULT_LANGUAGE


def set_language(code: str) -> None:
    global _lang
    _lang = code if code in LANGUAGES else DEFAULT_LANGUAGE


def get_language() -> str:
    return _lang


def t(key: str, **kw) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        return key
    s = entry.get(_lang) or entry["en"]
    return s.format(**kw) if kw else s


STRINGS: dict[str, dict[str, str]] = {
    # ── geral / barra lateral ─────────────────────────────────────────
    "language": {"en": "Language", "de": "Sprache", "pt": "Idioma"},
    "viewer": {"en": "viewer", "de": "Viewer", "pt": "visualizador"},
    "open_project": {"en": "📂 Open project", "de": "📂 Projekt öffnen", "pt": "📂 Abrir projeto"},
    "path_label": {"en": "Project folder or .tlproj file", "de": "Projektordner oder .tlproj-Datei",
                   "pt": "Pasta do projeto ou arquivo .tlproj"},
    "path_placeholder": {"en": r"C:\TextureLab\Resultados_v3  or  /data/.../Resultados_v3",
                         "de": r"C:\TextureLab\Resultados_v3  oder  /data/.../Resultados_v3",
                         "pt": r"C:\TextureLab\Resultados_v3  ou  /data/.../Resultados_v3"},
    "path_help": {"en": "Output folder of the pipeline (with projeto.json) or the .tlproj package. "
                        "Older folders (Resultados_v2) also open, with fewer plots.",
                  "de": "Ausgabeordner der Pipeline (mit projeto.json) oder das .tlproj-Paket. "
                        "Ältere Ordner (Resultados_v2) lassen sich auch öffnen, mit weniger Diagrammen.",
                  "pt": "Pasta de saída do pipeline (com projeto.json) ou o pacote .tlproj. "
                        "Também abre pastas antigas (Resultados_v2), com menos gráficos."},
    "open_btn": {"en": "Open folder / file", "de": "Ordner / Datei öffnen", "pt": "Abrir pasta / arquivo"},
    "upload_label": {"en": "…or upload a .tlproj", "de": "…oder .tlproj hochladen", "pt": "…ou envie um .tlproj"},
    "open_upload_btn": {"en": "Open uploaded package", "de": "Hochgeladenes Paket öffnen",
                        "pt": "Abrir pacote enviado"},
    "project_caption": {"en": "{ok}/{n} files ok · core {core} · format {fmt}",
                        "de": "{ok}/{n} Dateien ok · Kern {core} · Format {fmt}",
                        "pt": "{ok}/{n} arquivos ok · núcleo {core} · formato {fmt}"},
    "close_project": {"en": "Close project", "de": "Projekt schließen", "pt": "Fechar projeto"},
    "viewer_title": {"en": "Project viewer", "de": "Projekt-Viewer", "pt": "Visualizador de projetos"},
    "viewer_info": {"en": "Open a project computed on the server from the sidebar. Nothing is recomputed in this app.",
                    "de": "Öffnen Sie in der Seitenleiste ein auf dem Server berechnetes Projekt. "
                          "In dieser App wird nichts neu berechnet.",
                    "pt": "Abra um projeto calculado no servidor pela barra lateral. Nada é recalculado neste app."},

    # ── erros / avisos do projeto ─────────────────────────────────────
    "err_not_project": {"en": "'{path}' is neither a project folder nor a .tlproj file",
                        "de": "'{path}' ist weder ein Projektordner noch eine .tlproj-Datei",
                        "pt": "'{path}' não é uma pasta de projeto nem um arquivo .tlproj"},
    "err_bad_index": {"en": "projeto.json does not belong to a TextureLab project",
                      "de": "projeto.json gehört nicht zu einem TextureLab-Projekt",
                      "pt": "projeto.json não é de um projeto TextureLab"},
    "err_format_newer": {"en": "format {fmt} is newer than this app (supports up to {sup}): update the app",
                         "de": "Format {fmt} ist neuer als diese App (unterstützt bis {sup}): App aktualisieren",
                         "pt": "formato {fmt} é mais novo que este app (suporta até {sup}): atualize o app"},
    "err_no_results": {"en": "no resumo.json found: this does not look like a project",
                       "de": "keine resumo.json gefunden: das sieht nicht nach einem Projekt aus",
                       "pt": "nenhum resumo.json encontrado: isto não parece um projeto"},
    "err_open": {"en": "Could not open: {msg}", "de": "Öffnen nicht möglich: {msg}",
                 "pt": "Não foi possível abrir: {msg}"},
    "warn_no_index": {"en": "folder without projeto.json (old results): index built while reading",
                      "de": "Ordner ohne projeto.json (alte Ergebnisse): Index beim Lesen erstellt",
                      "pt": "pasta sem projeto.json (resultados antigos): índice montado na leitura"},
    "warn_mixed": {"en": "results from different core versions mixed: recompute with --skip-done",
                   "de": "Ergebnisse verschiedener Kernversionen gemischt: mit --skip-done neu berechnen",
                   "pt": "resultados de versões diferentes do núcleo misturados: recalcule com --skip-done"},
    "warn_old_core": {"en": "results computed with core {vers}: no Hurst and no viewer data (Abbott, profiles); "
                            "recompute on the server with the current version",
                      "de": "Ergebnisse mit Kern {vers} berechnet: kein Hurst und keine Viewer-Daten "
                            "(Abbott, Profile); auf dem Server mit der aktuellen Version neu berechnen",
                      "pt": "resultados calculados com núcleo {vers}: sem Hurst e sem dados de visualização "
                            "(Abbott, perfis); recalcule no servidor com a versão atual"},
    "warn_errors": {"en": "{n} file(s) failed in the computation", "de": "{n} Datei(en) mit Fehler in der Berechnung",
                    "pt": "{n} arquivo(s) com erro no cálculo"},

    # ── filtros e vistas ──────────────────────────────────────────────
    "filters": {"en": "🔎 Filters", "de": "🔎 Filter", "pt": "🔎 Filtros"},
    "all": {"en": "all", "de": "alle", "pt": "todos"},
    "n_after_filters": {"en": "{n} file(s) after filters", "de": "{n} Datei(en) nach Filtern",
                        "pt": "{n} arquivo(s) após filtros"},
    "no_files_filters": {"en": "No files match the current filters.", "de": "Keine Dateien für die aktuellen Filter.",
                         "pt": "Nenhum arquivo com os filtros atuais."},
    "view": {"en": "View", "de": "Ansicht", "pt": "Vista"},
    "view_summary": {"en": "📋 Summary", "de": "📋 Übersicht", "pt": "📋 Resumo"},
    "view_file": {"en": "🔎 File", "de": "🔎 Datei", "pt": "🔎 Arquivo"},
    "view_compare": {"en": "📊 Compare", "de": "📊 Vergleichen", "pt": "📊 Comparar"},
    "view_stats": {"en": "🧮 Statistics", "de": "🧮 Statistik", "pt": "🧮 Estatística"},
    "view_server": {"en": "🖥️ Compute on server", "de": "🖥️ Auf dem Server berechnen",
                    "pt": "🖥️ Calcular no servidor"},

    # ── agrupamentos / colunas de identificação ───────────────────────
    "grp_file": {"en": "File", "de": "Datei", "pt": "Arquivo"},
    "grp_section": {"en": "Road section", "de": "Strecke", "pt": "Trecho"},
    "grp_surface": {"en": "Surface course", "de": "Deckschicht", "pt": "Revestimento"},
    "grp_mp": {"en": "Measuring point (MP)", "de": "Messpunkt (MP)", "pt": "Ponto de medição (MP)"},
    "grp_section_surface": {"en": "Pavement (section + surface course)", "de": "Belag (Strecke + Deckschicht)",
                            "pt": "Superfície (trecho + revestimento)"},
    "std_caption": {"en": "std = standard deviation between the files of the group (– when the group has a single file).",
                    "de": "Std.-Abw. = Standardabweichung zwischen den Dateien der Gruppe (– bei nur einer Datei).",
                    "pt": "dp = desvio padrão entre os arquivos do grupo (– quando o grupo tem um só arquivo)."},
    "grp_section_mp": {"en": "Section + MP", "de": "Strecke + MP", "pt": "Trecho + MP"},
    "col_nr": {"en": "Run", "de": "Lauf", "pt": "NR"},
    "col_date": {"en": "Date", "de": "Datum", "pt": "Data"},

    # ── grupos de parâmetros / cadeias ────────────────────────────────
    "pg_chainA": {"en": "Chain A – ISO 13473-1", "de": "Kette A – ISO 13473-1", "pt": "Cadeia A – ISO 13473-1"},
    "pg_profile": {"en": "Profile (descriptive)", "de": "Profil (deskriptiv)", "pt": "Perfil (descritivo)"},
    "pg_hurst": {"en": "Hurst / fractal (descriptive)", "de": "Hurst / fraktal (deskriptiv)",
                 "pt": "Hurst / fractal (descritivo)"},
    "pg_SF": {"en": "Areal SF – ISO 25178 (S 0.05 mm, F plane)", "de": "Flächig SF – ISO 25178 (S 0,05 mm, F Ebene)",
              "pt": "Areal SF – ISO 25178 (S 0,05 mm, F plano)"},
    "pg_SL5": {"en": "Areal SL5 – ISO 25178 (+ L 5 mm)", "de": "Flächig SL5 – ISO 25178 (+ L 5 mm)",
               "pt": "Areal SL5 – ISO 25178 (+ L 5 mm)"},
    "pg_MICRO": {"en": "Areal MICRO (L 0.5 mm, provisional)", "de": "Flächig MICRO (L 0,5 mm, vorläufig)",
                 "pt": "Areal MICRO (L 0,5 mm, provisório)"},
    "ch_SF": {"en": "SF (S 0.05 mm, F plane)", "de": "SF (S 0,05 mm, F Ebene)", "pt": "SF (S 0,05 mm, F plano)"},
    "ch_SL5": {"en": "SL5 (+ L 5 mm)", "de": "SL5 (+ L 5 mm)", "pt": "SL5 (+ L 5 mm)"},
    "ch_MICRO": {"en": "MICRO (L 0.5 mm, provisional)", "de": "MICRO (L 0,5 mm, vorläufig)",
                 "pt": "MICRO (L 0,5 mm, provisório)"},

    # ── resumo ────────────────────────────────────────────────────────
    "params_by_file": {"en": "Parameters per file", "de": "Parameter je Datei", "pt": "Parâmetros por arquivo"},
    "param_groups": {"en": "Parameter groups", "de": "Parametergruppen", "pt": "Grupos de parâmetros"},
    "mean_by_group": {"en": "Mean per group", "de": "Mittelwert je Gruppe", "pt": "Média por grupo"},
    "group_by": {"en": "Group by", "de": "Gruppieren nach", "pt": "Agrupar por"},
    "agg_mean": {"en": "mean", "de": "Mittelwert", "pt": "média"},
    "agg_std": {"en": "std", "de": "Std.-Abw.", "pt": "dp"},
    "agg_count": {"en": "n", "de": "n", "pt": "n"},
    "dl_full_csv": {"en": "⬇️ Full table (CSV)", "de": "⬇️ Gesamttabelle (CSV)", "pt": "⬇️ Tabela completa (CSV)"},
    "dl_excel": {"en": "⬇️ Excel", "de": "⬇️ Excel", "pt": "⬇️ Excel"},
    "need_openpyxl": {"en": "Install openpyxl to export Excel.", "de": "openpyxl installieren, um Excel zu exportieren.",
                      "pt": "Instale openpyxl para exportar Excel."},

    # ── arquivo ───────────────────────────────────────────────────────
    "file": {"en": "File", "de": "Datei", "pt": "Arquivo"},
    "calc_error": {"en": "computation error", "de": "Berechnungsfehler", "pt": "erro no cálculo"},
    "file_caption": {"en": "{L:.0f} × {W:.0f} mm · dx {dx} mm · {n:.0f} M points · core {core} · recipe {rec}",
                     "de": "{L:.0f} × {W:.0f} mm · dx {dx} mm · {n:.0f} Mio. Punkte · Kern {core} · Rezept {rec}",
                     "pt": "{L:.0f} × {W:.0f} mm · dx {dx} mm · {n:.0f} M pontos · núcleo {core} · receita {rec}"},
    "no_view_data": {"en": "This result has no visualizacao.npz (old core): profiles, Abbott and histogram are not "
                           "available. Recompute on the server with the current version to see everything.",
                     "de": "Dieses Ergebnis hat keine visualizacao.npz (alter Kern): Profile, Abbott und Histogramm "
                           "sind nicht verfügbar. Auf dem Server mit der aktuellen Version neu berechnen.",
                     "pt": "Este resultado não tem visualizacao.npz (núcleo antigo): perfis, Abbott e histograma não "
                           "estão disponíveis. Recalcule no servidor com a versão atual para ver tudo."},
    "tab_surface": {"en": "Surface", "de": "Oberfläche", "pt": "Superfície"},
    "tab_profiles": {"en": "Profiles", "de": "Profile", "pt": "Perfis"},
    "tab_spectrum": {"en": "Spectrum", "de": "Spektrum", "pt": "Espectro"},
    "tab_abbott": {"en": "Abbott / heights", "de": "Abbott / Höhen", "pt": "Abbott / alturas"},
    "tab_psd": {"en": "PSD / Hurst", "de": "PSD / Hurst", "pt": "PSD / Hurst"},
    "tab_segments": {"en": "Segments", "de": "Segmente", "pt": "Segmentos"},
    "tab_all_params": {"en": "All parameters", "de": "Alle Parameter", "pt": "Todos os parâmetros"},
    "surface": {"en": "Surface", "de": "Oberfläche", "pt": "Superfície"},
    "surf_raw": {"en": "Raw (preview)", "de": "Roh (Vorschau)", "pt": "Bruta (prévia)"},
    "surf_sl5": {"en": "SL5 filtered", "de": "SL5 gefiltert", "pt": "SL5 filtrada"},
    "type": {"en": "Type", "de": "Typ", "pt": "Tipo"},
    "kind_map": {"en": "Map", "de": "Karte", "pt": "Mapa"},
    "kind_3d": {"en": "3D", "de": "3D", "pt": "3D"},
    "points_plot": {"en": "Points in plot", "de": "Punkte im Diagramm", "pt": "Pontos no gráfico"},
    "remove_plane": {"en": "Remove plane (display)", "de": "Ebene entfernen (Anzeige)", "pt": "Remover plano (exibição)"},
    "vert_exag": {"en": "Vertical exaggeration", "de": "Vertikale Überhöhung", "pt": "Exagero vertical"},
    "vert_exag_help": {"en": "1× = true scale (x, y and z in the same units). Higher values only stretch the drawing; "
                             "parameters are not affected.",
                       "de": "1× = maßstabsgetreu (x, y und z in gleichen Einheiten). Höhere Werte strecken nur die "
                             "Darstellung; die Parameter ändern sich nicht.",
                       "pt": "1× = escala real (x, y e z na mesma unidade). Valores maiores só esticam o desenho; "
                             "os parâmetros não mudam."},
    "remove_plane_help": {"en": "Only for the raw preview (the SL5 surface is already filtered).",
                          "de": "Nur für die Rohvorschau (die SL5-Oberfläche ist bereits gefiltert).",
                          "pt": "Só para a prévia bruta (a superfície SL5 já é filtrada)."},
    "no_preview": {"en": "No preview.", "de": "Keine Vorschau.", "pt": "Sem prévia."},
    "preview_caption": {"en": "Preview with step {step:.3f} mm (every {k} points); parameters were computed on the full grid.",
                        "de": "Vorschau mit Schrittweite {step:.3f} mm (jeder {k}. Punkt); die Parameter wurden auf dem "
                              "vollständigen Raster berechnet.",
                        "pt": "Prévia com passo {step:.3f} mm (a cada {k} pontos); os parâmetros foram calculados na "
                              "grade completa."},
    "ax_road": {"en": "driving direction [mm]", "de": "Fahrtrichtung [mm]", "pt": "via [mm]"},
    "ax_width": {"en": "width [mm]", "de": "Breite [mm]", "pt": "largura [mm]"},
    "ax_height": {"en": "height [mm]", "de": "Höhe [mm]", "pt": "altura [mm]"},
    "ax_mr": {"en": "Material ratio [%]", "de": "Materialanteil [%]", "pt": "Material ratio [%]"},
    "ax_density": {"en": "density [1/mm]", "de": "Dichte [1/mm]", "pt": "densidade [1/mm]"},
    "prof_clean": {"en": "y={y:.1f} mm cleaned", "de": "y={y:.1f} mm bereinigt", "pt": "y={y:.1f} mm limpo"},
    "prof_lp": {"en": "y={y:.1f} mm low-pass", "de": "y={y:.1f} mm Tiefpass", "pt": "y={y:.1f} mm passa-baixa"},
    "prof_title": {"en": "Chain A: 0.5 mm profiles (cleaned and after low-pass); lines = 100 mm segments",
                   "de": "Kette A: 0,5-mm-Profile (bereinigt und nach Tiefpass); Linien = 100-mm-Segmente",
                   "pt": "Cadeia A: perfis de 0,5 mm (limpos e após passa-baixa); linhas = segmentos de 100 mm"},
    "native_title": {"en": "Line at native resolution (dx {dx} mm, y = {y:.1f} mm)",
                     "de": "Linie in nativer Auflösung (dx {dx} mm, y = {y:.1f} mm)",
                     "pt": "Linha na resolução nativa (dx {dx} mm, y = {y:.1f} mm)"},
    "no_profiles": {"en": "No profiles in this result.", "de": "Keine Profile in diesem Ergebnis.",
                    "pt": "Sem perfis neste resultado."},
    "mean_sd": {"en": "mean ± sd", "de": "Mittelwert ± Std.-Abw.", "pt": "média ± dp"},
    "spec_title": {"en": "Texture spectrum – ISO 13473-4 (method 1)", "de": "Texturspektrum – ISO 13473-4 (Verfahren 1)",
                   "pt": "Espectro de textura – ISO 13473-4 (método 1)"},
    "ax_lambda_band": {"en": "λ one-third-octave band centre [mm] — log scale (grid: 1–9 per decade)",
                       "de": "λ Terzband-Mittenwellenlänge [mm] — log. Skala (Raster: 1–9 je Dekade)",
                       "pt": "λ centro do terço de oitava [mm] — escala log (grade: 1–9 por década)"},
    "display": {"en": "Display", "de": "Darstellung", "pt": "Exibição"},
    "mode_overlay": {"en": "Overlaid", "de": "Überlagert", "pt": "Sobrepostas"},
    "mode_side": {"en": "Side by side (one per chain)", "de": "Nebeneinander (je Kette)",
                  "pt": "Lado a lado (uma por cadeia)"},
    "mode_grid": {"en": "One per plot", "de": "Eine je Diagramm", "pt": "Uma por gráfico"},
    "abbott_title": {"en": "Abbott-Firestone curve (heights relative to mean)",
                     "de": "Abbott-Firestone-Kurve (Höhen bezogen auf den Mittelwert)",
                     "pt": "Curva de Abbott-Firestone (alturas em relação à média)"},
    "pdf_title": {"en": "Height distribution (PDF)", "de": "Höhenverteilung (PDF)", "pt": "Distribuição de alturas (PDF)"},
    "abbott_caption_file": {"en": "Dashed: ISO 13565-2 equivalent straight line (0 % to 100 %); diamonds: Smr1 and Smr2. "
                                  "Slope = −Sk/100 % [mm/%].",
                            "de": "Gestrichelt: äquivalente Gerade nach ISO 13565-2 (0 % bis 100 %); Rauten: Smr1 und "
                                  "Smr2. Steigung = −Sk/100 % [mm/%].",
                            "pt": "Tracejado: reta equivalente da ISO 13565-2 (de 0 % a 100 %); losangos: Smr1 e Smr2. "
                                  "Inclinação = −Sk/100 % [mm/%]."},
    "no_abbott": {"en": "No Abbott curves in this result.", "de": "Keine Abbott-Kurven in diesem Ergebnis.",
                  "pt": "Sem curvas de Abbott neste resultado."},
    "core_top": {"en": "core top", "de": "Kernoberkante", "pt": "topo do núcleo"},
    "slope": {"en": "slope", "de": "Steigung", "pt": "inclinação"},
    "col_slope": {"en": "slope [mm/%]", "de": "Steigung [mm/%]", "pt": "inclinação [mm/%]"},
    "psd_mean": {"en": "mean PSD", "de": "mittlere PSD", "pt": "PSD média"},
    "psd_title": {"en": "1D PSD in driving direction (Welch) and Hurst fit – descriptive",
                  "de": "1D-PSD in Fahrtrichtung (Welch) und Hurst-Anpassung – deskriptiv",
                  "pt": "PSD 1D no sentido da via (Welch) e ajuste de Hurst – descritivo"},
    "psd_caption": {"en": "Dashed lines: slope fitted on the server (PSD ∝ λ^β), drawn through the geometric mean of the band.",
                    "de": "Gestrichelt: auf dem Server angepasste Steigung (PSD ∝ λ^β), durch das geometrische Mittel "
                          "des Bereichs gezeichnet.",
                    "pt": "Retas tracejadas: inclinação ajustada no servidor (PSD ∝ λ^β), desenhadas pela média "
                          "geométrica da faixa."},
    "no_psd": {"en": "No PSD in this result (core before v3).", "de": "Keine PSD in diesem Ergebnis (Kern vor v3).",
               "pt": "Sem PSD neste resultado (núcleo anterior à v3)."},
    "msd_hist_title": {"en": "MSD per 100 mm segment (valid)", "de": "MSD je 100-mm-Segment (gültig)",
                       "pt": "MSD por segmento de 100 mm (válidos)"},
    "msd_pos_title": {"en": "MSD across the width", "de": "MSD über die Breite", "pt": "MSD ao longo da largura"},
    "ax_pos_width": {"en": "position across width [mm]", "de": "Position in der Breite [mm]",
                     "pt": "posição na largura [mm]"},
    "segment": {"en": "segment", "de": "Segment", "pt": "segmento"},
    "param": {"en": "parameter", "de": "Parameter", "pt": "parâmetro"},
    "value": {"en": "value", "de": "Wert", "pt": "valor"},
    "recipe_expander": {"en": "Recipe (core version, configuration, LAZ)", "de": "Rezept (Kernversion, Konfiguration, LAZ)",
                        "pt": "Receita (versão do núcleo, configuração, LAZ)"},

    # ── comparar ──────────────────────────────────────────────────────
    "compare_by": {"en": "Compare by", "de": "Vergleichen nach", "pt": "Comparar por"},
    "groups": {"en": "Groups", "de": "Gruppen", "pt": "Grupos"},
    "pick_group": {"en": "Choose at least one group.", "de": "Mindestens eine Gruppe wählen.",
                   "pt": "Escolha pelo menos um grupo."},
    "parameters": {"en": "Parameters", "de": "Parameter", "pt": "Parâmetros"},
    "chart": {"en": "Chart", "de": "Diagramm", "pt": "Gráfico"},
    "chart_bar": {"en": "Bars (mean ± sd)", "de": "Balken (Mittelwert ± Std.-Abw.)", "pt": "Barras (média ± dp)"},
    "chart_box": {"en": "Box (all files)", "de": "Boxplot (alle Dateien)", "pt": "Caixa (todos os arquivos)"},
    "spectra_title": {"en": "One-third-octave spectra", "de": "Terzbandspektren", "pt": "Espectros de terço de oitava"},
    "ltx_by_band": {"en": "**L_tx [dB re 1 µm] per band – mean per group**",
                    "de": "**L_tx [dB re 1 µm] je Terzband – Mittelwert je Gruppe**",
                    "pt": "**L_tx [dB ref. 1 µm] por banda – média por grupo**"},
    "all_samples": {"en": "All samples ({n})", "de": "Alle Proben ({n})", "pt": "Todas as amostras ({n})"},
    "dl_spectra": {"en": "⬇️ Spectra (CSV)", "de": "⬇️ Spektren (CSV)", "pt": "⬇️ Espectros (CSV)"},
    "abbott_curves": {"en": "Abbott-Firestone curves", "de": "Abbott-Firestone-Kurven", "pt": "Curvas de Abbott-Firestone"},
    "areal_chain": {"en": "Areal chain", "de": "Flächige Kette", "pt": "Cadeia areal"},
    "show_rk": {"en": "Equivalent line and Smr1/Smr2 (ISO 13565-2)", "de": "Äquivalente Gerade und Smr1/Smr2 (ISO 13565-2)",
                "pt": "Reta equivalente e Smr1/Smr2 (ISO 13565-2)"},
    "zoom_core": {"en": "Zoom on core (heights between 1 % and 99 %)", "de": "Kernbereich zoomen (Höhen zwischen 1 % und 99 %)",
                  "pt": "Zoom no núcleo (alturas entre 1 % e 99 %)"},
    "columns": {"en": "Columns", "de": "Spalten", "pt": "Colunas"},
    "legend_hint": {"en": "Click a legend item to hide the curve; double-click to show only that one.",
                    "de": "Legendeneintrag anklicken blendet die Kurve aus; Doppelklick zeigt nur diese.",
                    "pt": "Clique em um item da legenda para esconder a curva; duplo clique para ver só ela."},
    "abbott_caption_cmp": {"en": "Curve = mean of the group's curves; dashed = equivalent line; diamonds = Smr1 and Smr2 "
                                 "(hover for values). Smr1, Smr2 and Sk are group means, computed per file on the server. "
                                 "Equivalent line slope = −Sk/100 % [mm/%].",
                           "de": "Kurve = Mittel der Kurven der Gruppe; gestrichelt = äquivalente Gerade; Rauten = Smr1 "
                                 "und Smr2 (Werte per Mouseover). Smr1, Smr2 und Sk sind Gruppenmittel, je Datei auf dem "
                                 "Server berechnet. Steigung der äquivalenten Geraden = −Sk/100 % [mm/%].",
                           "pt": "Curva = média das curvas do grupo; tracejado = reta equivalente; losangos = Smr1 e Smr2 "
                                 "(passe o mouse para ver os valores). Smr1, Smr2 e Sk são médias do grupo, calculados por "
                                 "arquivo no servidor. Inclinação da reta equivalente = −Sk/100 % [mm/%]."},
    "sk_family_mean": {"en": "**Sk family – mean per group**", "de": "**Sk-Familie – Mittelwert je Gruppe**",
                       "pt": "**Família Sk – média por grupo**"},
    "dl_sk": {"en": "⬇️ Sk family (CSV)", "de": "⬇️ Sk-Familie (CSV)", "pt": "⬇️ Família Sk (CSV)"},
    "no_abbott_project": {"en": "No Abbott curves in the project (results of the old core).",
                          "de": "Keine Abbott-Kurven im Projekt (Ergebnisse des alten Kerns).",
                          "pt": "Sem curvas de Abbott no projeto (resultados do núcleo antigo)."},

    # ── estatística ───────────────────────────────────────────────────
    "color_by": {"en": "Colour by", "de": "Farbe nach", "pt": "Cor por"},
    "pick_two": {"en": "Choose at least two parameters.", "de": "Mindestens zwei Parameter wählen.",
                 "pt": "Escolha pelo menos dois parâmetros."},
    "corr_title": {"en": "Correlation (Pearson)", "de": "Korrelation (Pearson)", "pt": "Correlação (Pearson)"},
    "scatter_title": {"en": "Scatter", "de": "Streudiagramm", "pt": "Dispersão"},
    "pca_title": {"en": "PCA (standardised parameters)", "de": "PCA (standardisierte Parameter)",
                  "pt": "PCA (parâmetros padronizados)"},
    "pca_few": {"en": "Too few complete files for PCA.", "de": "Zu wenige vollständige Dateien für PCA.",
                "pt": "Poucos arquivos completos para PCA."},
    "variance": {"en": "variance [%]", "de": "Varianz [%]", "pt": "variância [%]"},
    "explained_var": {"en": "Explained variance", "de": "Erklärte Varianz", "pt": "Variância explicada"},
    "pca_dropped": {"en": "{n} file(s) missing some parameter were left out of the PCA.",
                    "de": "{n} Datei(en) mit fehlenden Parametern wurden nicht in die PCA aufgenommen.",
                    "pt": "{n} arquivo(s) sem algum dos parâmetros ficaram fora da PCA."},

    # ── servidor ──────────────────────────────────────────────────────
    "server_title": {"en": "Compute or recompute on the server", "de": "Auf dem Server berechnen oder neu berechnen",
                     "pt": "Calcular ou recalcular no servidor"},
    "server_intro": {"en": "This app does not compute. Anything missing (new files, another configuration, a newer core) "
                           "is computed on the server by the pipeline, and the result is opened here.",
                     "de": "Diese App rechnet nicht. Was fehlt (neue Dateien, andere Konfiguration, neuerer Kern), wird "
                           "auf dem Server von der Pipeline berechnet und das Ergebnis hier geöffnet.",
                     "pt": "Este app não calcula. O que faltar (arquivos novos, outra configuração, núcleo mais novo) "
                           "é calculado no servidor pelo pipeline, e o resultado é aberto aqui."},
    "config_used": {"en": "Configuration used in this project", "de": "In diesem Projekt verwendete Konfiguration",
                    "pt": "Configuração usada neste projeto"},
    "file_status": {"en": "File status", "de": "Status der Dateien", "pt": "Situação dos arquivos"},
    "server_help": {
        "en": """
**Workflow:** the server computes once, the app only opens and draws.

1. Copy the LAZ files to the server (folder `LAZ/`).
2. On the server, run the batch with controlled resource use (CPU, RAM and disk limited; see `pipeline/README.md`):
```bash
cd /data/callai/workspace/tyron/texturelab_repo/pipeline
./servidor.sh iniciar /data/callai/workspace/tyron/LAZ /data/callai/workspace/tyron/Resultados_v3
./servidor.sh status        # also: pausar (pause), continuar (resume), parar (stop), log
```
3. Copy `Resultados_v3.tlproj` (or the folder `Resultados_v3`) to the PC, via `scp` or Nextcloud:
```bash
scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
```
4. Open the `.tlproj` or the folder here.

`--skip-done` only recomputes what changed (core version, configuration or the LAZ itself).
Another configuration: `EXTRA="--config settings.json" ./servidor.sh iniciar LAZ Resultados_other` (e.g. `{"A_lp_design_mm": 2.4}`).
""",
        "de": """
**Ablauf:** Der Server rechnet einmal, die App öffnet und zeichnet nur.

1. LAZ-Dateien auf den Server kopieren (Ordner `LAZ/`).
2. Auf dem Server den Stapellauf mit begrenzter Ressourcennutzung starten (CPU, RAM und Festplatte begrenzt; siehe `pipeline/README.md`):
```bash
cd /data/callai/workspace/tyron/texturelab_repo/pipeline
./servidor.sh iniciar /data/callai/workspace/tyron/LAZ /data/callai/workspace/tyron/Resultados_v3
./servidor.sh status        # außerdem: pausar (pausieren), continuar (fortsetzen), parar (stoppen), log
```
3. `Resultados_v3.tlproj` (oder den Ordner `Resultados_v3`) per `scp` oder Nextcloud auf den PC kopieren:
```bash
scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
```
4. Die `.tlproj`-Datei oder den Ordner hier öffnen.

`--skip-done` berechnet nur neu, was sich geändert hat (Kernversion, Konfiguration oder die LAZ-Datei selbst).
Andere Konfiguration: `EXTRA="--config einstellungen.json" ./servidor.sh iniciar LAZ Resultados_andere` (z. B. `{"A_lp_design_mm": 2.4}`).
""",
        "pt": """
**Fluxo:** o servidor calcula uma vez, o app só abre e desenha.

1. Copie os LAZ para o servidor (pasta `LAZ/`).
2. No servidor, rode o lote com uso controlado (CPU, RAM e disco limitados; ver `pipeline/README.md`):
```bash
cd /data/callai/workspace/tyron/texturelab_repo/pipeline
./servidor.sh iniciar /data/callai/workspace/tyron/LAZ /data/callai/workspace/tyron/Resultados_v3
./servidor.sh status        # também: pausar, continuar, parar, log
```
3. Copie `Resultados_v3.tlproj` (ou a pasta `Resultados_v3`) para o PC, por `scp` ou Nextcloud:
```bash
scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
```
4. Abra aqui o `.tlproj` ou a pasta.

`--skip-done` só recalcula o que mudou (versão do núcleo, configuração ou o próprio LAZ).
Para outra configuração: `EXTRA="--config ajustes.json" ./servidor.sh iniciar LAZ Resultados_outra` (ex.: `{"A_lp_design_mm": 2.4}`).
""",
    },
}
