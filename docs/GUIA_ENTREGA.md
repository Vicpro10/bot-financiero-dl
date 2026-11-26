# Guía de entrega (lista de comprobación)

Entrega: **mañana 10:00**. Requisitos de la tarea y dónde se cumple cada uno:

| # | Requisito de la tarea | Dónde está | Estado |
|---|---|---|---|
| 1 | Presentación ejecutiva | Deck de Slides (exportar a PDF o PPTX desde Share › Export) | Llenar las celdas `[__]` con los resultados de Colab |
| 2 | Git con el código | Este repositorio: commits y etiquetas `v0.1` a `v0.9` | Falta subir a GitHub |
| 3 | README completísimo con versiones y avances | `README.md` (secciones 10 y 5.2) | Listo; actualizar la sección 9 con los resultados de la T4 |
| 4 | Datos reales de Polymarket o MEXC | MEXC: `/api/market` y el panel (precio en vivo y velas cerradas) | Listo |
| 5 | Operaciones y PnL | Panel: tabla *Operaciones Paper y PnL*, curva de patrimonio, comparación de corridas | Listo |

## Pasos que solo tú puedes hacer
1. **Colab con T4:** sube `notebook/DL_Finanzas_BTCUSDT_MEXC.ipynb`, elige *GPU T4* y pulsa *Ejecutar todo* (tarda unos 2 minutos). Guarda el notebook con sus salidas (*Archivo › Guardar una copia en Drive*) y comparte el enlace con permiso de lectura. Descarga `entrega_dl_finanzas.zip`.
2. **Integrar el modelo de la T4:** copia `modelos/dl_best/` del zip a `backend/models/current/` y reinicia el backend.
3. **GitHub:** crea un repositorio vacío y súbelo con `git remote add origin <url>` y `git push -u origin main --tags`.
4. **Demostración:** levanta el backend, Ollama y el frontend. Corre un replay *Modelo sin LLM* y otro *Agente con LLM* con los mismos días. Provoca 2 o 3 errores con los botones del panel y toma capturas.

## Para la exposición
- El bot **no opera con dinero real**: es modo Paper sobre datos reales de MEXC, como pide el protocolo.
- Explica por qué TE se abre una sola vez y qué pasa si se ajusta el sistema después de verlo.
- El "+20%": reporta las cifras reales. La rúbrica dice que no se exige ganancia.
