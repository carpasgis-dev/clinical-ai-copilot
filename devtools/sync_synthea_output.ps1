<#
.SYNOPSIS
  Copia salida de Synthea (FHIR/CSV) al copilot si aún generas en synthea/output local.

.DESCRIPTION
  Por defecto el repo Synthea queda configurado para escribir en
  clinical-ai-copilot/data/synthea/output/ (ver synthea.properties).
  Usa este script solo si tienes datos antiguos en synthea/output y quieres traerlos.

  Ejecutar desde clinical-ai-copilot (o cualquier cwd): powershell -File scripts/sync_synthea_output.ps1
#>
$ErrorActionPreference = "Stop"
$copilotRoot = Split-Path -Parent $PSScriptRoot
$syntheaOut = Join-Path $copilotRoot "..\synthea\output"
$destRoot = Join-Path $copilotRoot "data\synthea\output"

if (-not (Test-Path $syntheaOut)) {
    Write-Error "No existe la carpeta origen: $syntheaOut"
}
New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
Copy-Item -Path (Join-Path $syntheaOut "*") -Destination $destRoot -Recurse -Force
Write-Host "Copiado: $syntheaOut -> $destRoot"
