# Backups y restauración (RF-SRV-08)

## Qué se respalda

| Qué | Dónde vive | Cómo se respalda |
|---|---|---|
| Metadata: proyectos, versiones de datos, runs, usuarios, roles, auditoría | PostgreSQL, base `perceptron` | `pg_dump -Fc` |
| Tracking de MLflow | PostgreSQL, base `mlflow` | `pg_dump -Fc` |
| Datasets, runs, modelos, exportaciones | volumen `workspace` (`/data`) | `tar.gz` |
| Artefactos de MLflow | volumen `mlartifacts` | `tar.gz` |
| Claves de proveedores LLM y fuentes (cifradas) | `workspace` (archivo cifrado con `PERCEPTRON_MASTER_KEY`) | con el workspace |

La cola (Valkey) no se respalda: si se reinicia, solo se pierden los estudios que estaban en espera.

**`PERCEPTRON_MASTER_KEY` y `PERCEPTRON_SECRET_KEY` se guardan aparte**, en el gestor de secretos de la empresa. Sin la clave maestra, los secretos cifrados del backup no se pueden leer.

## Docker Compose

```bash
server/deploy/backup.sh /ruta/backups          # → /ruta/backups/perceptron-AAAAMMDDTHHMMSSZ/
server/deploy/restore.sh /ruta/backups/perceptron-AAAAMMDDTHHMMSSZ
```

- **Backup:** `backup.sh` detiene el servidor y los workers mientras copia, para que la base y los archivos queden coherentes, y los vuelve a levantar. Deja `SHA256SUMS`.
- **Restauración:** `restore.sh` verifica los checksums y **reemplaza** la base y los volúmenes.
- **Verificación en CI:** el job `compose` hace backup, borra todo (`down -v`), levanta vacío, restaura y comprueba que vuelven el proyecto y sus runs.

Programación sugerida: diaria, por cron del host, y retención de 7 diarios y 4 semanales:

```cron
30 2 * * * /opt/perceptron/server/deploy/backup.sh /backups >> /var/log/perceptron-backup.log 2>&1
```

## Kubernetes

- **PostgreSQL:** los backups del servicio gestionado (PITR) o `pg_dump` desde un `CronJob`.
- **Volúmenes:** snapshots de los PVC (`VolumeSnapshot`) del workspace y de los artefactos de MLflow, o copia con Velero.
- **Orden:** base primero y después los volúmenes, con el servidor escalado a 0 durante la copia.
