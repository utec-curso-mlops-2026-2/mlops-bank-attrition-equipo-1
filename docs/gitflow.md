# R1 · Git / GitHub: configuración paso a paso

1. **Repo desde plantilla**: en `utec-curso-mlops-2026-2/mlops-bank-attrition-template` → *Use this template* →
   owner `utec-curso-mlops-2026-2`, nombre `mlops-bank-attrition-equipo-1` .
   Invitar a Renzo y Fabrizio con permiso *Write*.
2. **Ramas**: `master` (producción), `develop` (integración), `feature/<descripcion>` (trabajo).
   ```bash
   git clone <url> && cd <repo>
   git checkout -b develop && git push -u origin develop
   git checkout -b feature/pipeline-entrenamiento develop
   ```
3. **Protección** (Settings → Branches / Rules) para `master` y `develop`:
   *Require a pull request before merging* (1 aprobación), *Require status checks* (`Instalación, lint y pruebas`),
   *Block force pushes*. Equivalente por CLI (si tienes permisos de admin):
   ```bash
   gh api -X PUT repos/<org>/<repo>/branches/develop/protection --input - <<'JSON'
   {"required_status_checks":{"strict":true,"contexts":["Instalación, lint y pruebas"]},
    "enforce_admins":false,
    "required_pull_request_reviews":{"required_approving_review_count":1},
    "restrictions":null}
   JSON
   ```
   Repetir con `master`. Si el plan del repo no permite reglas, documentarlo en `docs/decisiones.md`.
4. **Commits** atómicos con Conventional Commits: `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`.
5. **PRs**: `feature/*` → `develop` (revisado por otro integrante) y `develop` → `master` (release). Todos los integrantes
   deben tener commits y PRs propios: repartan al menos un PR por persona (p. ej. Renzo revisa R3, Fabrizio revisa R5).
6. `.gitignore` ya excluye datos, modelos serializados, entornos virtuales, `mlruns/` y credenciales; no lo debiliten.

## PRs sugeridos para esta primera parte
| Rama | Contenido |
|---|---|
| `feature/estructura-y-config` | config.yaml, data.py, features.py, tests de features |
| `feature/pipeline-entrenamiento` | models, metrics, train |
| `feature/mlflow-registry` | tracking, pyfunc_model, registry, tests e2e |
| `feature/docs-r1-r5` | README, runbooks, contrato de modelo |
