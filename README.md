# Despliegue Web Estático (GitHub Pages) + Dataset Hub NTP 900.058 — Ciro v3 (UPeU)

Este directorio (`deploy_github_pages/`) es un repositorio autocontenido listo para publicar en **GitHub** y activar **GitHub Pages** sin servidor backend.

## Contenido del Proyecto Unificado
- **`index.html` + `app.js` + `styles.css`**: Aplicación web estática que ejecuta los modelos **`models/ciro_cls_ntp_v3.onnx`** (`21.3 MB`) y **`models/yolo26s_ntp_7c_v3.onnx`** (`38.1 MB`) directamente en el navegador mediante **ONNX Runtime Web (WebAssembly)**.
- **`dataset/index.html`**: Portal interactivo del **Dataset Curado NTP 900.058 (`14,189` imágenes, `20,739` cajas)** con tabla de URLs oficiales, manifiesto criptográfico `dataset_manifest.json`, distribución `class_distribution.csv`, taxonomía `taxonomy_ntp900058.yaml` y muestras reales.
- **`dossier.html`**: Dossier técnico interactivo con la evaluación completa de los modelos mejorados Ciro v3.

## Pasos para Publicar en GitHub Pages
```bash
cd deploy_github_pages
git init
git add .
git commit -m "Deploy Ciro v3 ONNX Web App + Dataset Hub NTP 900.058"
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/<TU_REPOSITORIO>.git
git push -u origin main
```
Luego, en GitHub: **Settings → Pages → Source: `Deploy from a branch` → Branch: `main` / `(root)` → Save**.