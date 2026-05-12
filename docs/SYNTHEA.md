# Synthea y base clínica SQLite

## Qué es Synthea

[Synthea](https://github.com/synthetichealth/synthea) genera **poblaciones sintéticas** (no son pacientes reales). Salida por defecto: **FHIR / C-CDA** bajo `output/`. El **CSV** y otros exportadores hay que activarlos en `src/main/resources/synthea.properties` (ver [wiki](https://github.com/synthetichealth/synthea/wiki)).

**Importante:** el comando `./run_synthea` **no crea un `synthea.db` SQLite** listo para el copiloto. Ese fichero sería un paso **posterior** (importar CSV/FHIR a un esquema SQLite que definas tú o un ETL) y luego apuntar `CLINICAL_DB_PATH` ahí.

## Requisitos

- **JDK 17+** (LTS recomendado; ver README de Synthea).

## Clonar y generar población (Linux / macOS / Git Bash)

```bash
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build check test   # opcional, primera vez
./run_synthea -p 100
```

Argumentos útiles: `-p` población, `Estado [Ciudad]`, `-s` semilla, `-h` ayuda.

## Windows (PowerShell)

No existe `./run_synthea` nativo; equivale a pasar parámetros a Gradle:

```powershell
cd synthea
.\gradlew.bat run -Params="['-p','100']"
```

Con estado (ejemplo):

```powershell
.\gradlew.bat run -Params="['-p','100','Massachusetts']"
```

Si instalas **Git Bash**, puedes usar `./run_synthea -p 100` desde `bash` dentro del repo.

## Salida dentro del repo `clinical-ai-copilot` (recomendado en este workspace)

En `synthea/src/main/resources/synthea.properties` queda configurado (en tu clon local):

- `exporter.baseDirectory = ../clinical-ai-copilot/data/synthea/output/`
- `exporter.csv.export = true`

Así, al ejecutar Gradle **desde la carpeta `synthea/`** (hermana de `clinical-ai-copilot/` bajo `cursos_actividades/`), FHIR y CSV se escriben en:

`clinical-ai-copilot/data/synthea/output/`  
(subcarpetas típicas: `fhir/`, `csv/`, etc.)

Esa carpeta está en **`.gitignore`** del copilot (volumen grande).

### Si ya tienes datos viejos en `synthea/output/`

Script de copia (PowerShell), ejecutado desde la raíz del copilot:

```powershell
cd clinical-ai-copilot
powershell -ExecutionPolicy Bypass -File scripts/sync_synthea_output.ps1
```

## Activar CSV (para ETL a SQLite)

Con `exporter.csv.export = true`, tras cada run aparecen CSV bajo `data/synthea/output/csv/` (nombres según [wiki CSV](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary)).

En el copilot, variable de referencia: `SYNTHEA_CSV_DIR=data/synthea/output/csv` en `.env.example`.

## ETL → SQLite (`synthea.db`)

Desde la raíz de `clinical-ai-copilot`, con CSV ya generados:

```bash
python scripts/synthea_csv_to_sqlite.py
```

Opciones explícitas:

```bash
python scripts/synthea_csv_to_sqlite.py --csv-dir data/synthea/output/csv --db data/clinical/synthea.db
```

Equivale a leer `SYNTHEA_CSV_DIR` y `CLINICAL_DB_PATH` del entorno si no pasas flags.

Tablas creadas: `patients`, `conditions`, `medications` (columnas TEXT importadas desde los CSV).

## Conectar el copiloto

1. Genera CSV con Synthea y ejecuta el ETL anterior (p. ej. `data/clinical/synthea.db`).
2. En `.env`: `CLINICAL_DB_PATH=data/clinical/synthea.db` (o ruta absoluta).
3. Reinicia la API: el grafo usa `SqliteClinicalCapability` en la ruta **SQL** (conteo de vivos) y en **HYBRID** (`extract_clinical_summary`). Si el fichero no existe, se mantienen los **stubs** (misma semántica que en CI).

`GET /health` expone `clinical_db_loaded` y `clinical_db_path` (resuelto o `(unset or missing file)`).

## Referencia en el repo

- Contrato: `app/capabilities/contracts.py` → `ClinicalCapability`.
- Implementación SQLite de solo lectura: `app/capabilities/clinical_sql/sqlite_clinical_capability.py` → `SqliteClinicalCapability`.
