# Dataset — PAP Operator Turns

Extracted from 10 clinical PAP scripts. 364 operator turns with ABCDE phase weak labels and multi-label verbal act heuristics.

## Columns

| Column | Type | Description |
|--------|------|-------------|
| turno_id | str | Unique turn ID (CaseName_NNNN) |
| caso | str | Case name (Camila, Javiera, ...) |
| split | str | "train" or "held_out" |
| fase_abcde | str | Weak label from section header (A-E) |
| seccion | str | Section number within the case |
| is_ramificacion | bool | True if inside a RAMIFICACIÓN block |
| texto_operador | str | Operator turn text |
| contexto_previo | str | Previous patient turn (context) |
| actos_verbales | str | Pipe-separated verbal act labels |

## Split
- Train: 8 cases (291 turns)
- Held-out: Mercedes + Luis (73 turns)
