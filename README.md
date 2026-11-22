# ClipGod

> Transforme les streams et VODs **GTA 6** en clips verticaux (1080x1920) prêts pour les campagnes de clipping
> (Whop Content Rewards et équivalents). Aucun SaaS payant : tout est open source, et le calcul lourd tourne
> sur le **GPU T4 gratuit** de Google Colab ou Kaggle.

**Statut :** squelette. Les commandes existent mais lèvent `NotImplementedError` ; elles sont implémentées
module par module (voir [Feuille de route](#feuille-de-route)).

---

## Architecture : deux environnements, un seul lien (Google Drive)

```
 ┌──────────────── LAPTOP (CPU modeste, pas de GPU) ────────────────┐        ┌────── CLOUD GRATUIT (Colab / Kaggle, GPU T4) ──────┐
 │                                                                  │        │                                                    │
 │  clipgod discover ──► data/sources.json                          │        │  notebooks/clipgod_colab.ipynb                     │
 │  clipgod prepare  ──► upload ───────────┐                        │        │    1. monte Drive / lit /kaggle/working            │
 │  clipgod sync     ◄── download ──────┐  │                        │        │    2. prend le 1er job de ClipGod/jobs/            │
 │  clipgod monetize (validation)       │  │                        │        │    3. modules.scan.scan()   (faster-whisper GPU)   │
 │  clipgod publish  ──► YouTube/TikTok…│  │                        │        │    4. modules.render.render() (OpenCV + ffmpeg)    │
 │  clipgod learn    ──► data/model.pkl │  │                        │        │    5. écrit moments/, clips/, jobs_done/           │
 │  clipgod watch    (optionnel)        │  │                        │        │                                                    │
 └──────────────────────────────────────┼──┼────────────────────────┘        └───────────────────────▲────────────────────────────┘
                                        │  ▼                                                         │
                              ┌─────────┴──────────────── Google Drive : ClipGod/ ───────────────────┴──────┐
                              │  jobs/       <job_id>.json      ← écrit par le laptop, lu par le notebook   │
                              │  sources/    <video_id>.mp4     ← vidéo source uploadée par le laptop       │
                              │  moments/    <video_id>.json    ← écrit par le notebook                     │
                              │  clips/      <video_id>_<i>.mp4 ← écrit par le notebook, téléchargé par sync │
                              │  jobs_done/  <job_id>.json      ← marqueur de fin écrit par le notebook     │
                              └──────────────────────────────────────────────────────────────────────────────┘
```

Aucun serveur à maintenir, aucun port à ouvrir : le laptop dépose un job, le notebook le traite, le laptop récupère.

| Module | Où | Rôle |
|---|---|---|
| `discover` | laptop | YouTube Data API v3, Twitch Helix, Kick (scraping léger). Live ou VOD < 24h, > 45 min. |
| `watch` | laptop / VPS | Buffer roulant 8 min (streamlink) + score de hype du chat. **Désactivable** (`disable-watch`). |
| `scan` | **cloud GPU** | faster-whisper + signaux audio/texte/visuels/most-replayed → top moments + hooks. |
| `render` | **cloud GPU** | Recadrage YuNet, layout gameplay 70% / facecam 30%, sous-titres mot à mot, -14 LUFS. |
| `publish` | laptop | Max 6 posts/jour, créneaux EU matin / US soir, A/B test des hooks, quota YouTube. |
| `monetize` | laptop | Scraping Whop, classement $/h, validation des clips (→ `data/rejected.json`). |
| `learn` | laptop | Vues à 1h/6h/24h/72h → GradientBoostingRegressor → `data/model.pkl` (relu par `scan`). |

> `modules/scan.py` et `modules/render.py` sont **importables à l'identique** depuis le notebook et depuis
> le CLI (pas de `if __name__ == "__main__"`, imports lourds paresseux).

---

## Arborescence

```
clipgod/
├── clipgod.py              # CLI Typer (point d'entrée)
├── config.yaml             # configuration unique (laptop + cloud)
├── pyproject.toml          # deps locales ; extras [cloud], [watch], [dev]
├── .env.example            # variables d'environnement attendues (clés API)
├── modules/                # une étape du pipeline = un module
│   ├── discover.py  watch.py  scan.py  render.py
│   └── publish.py   monetize.py  learn.py
├── core/                   # briques communes (CLI + notebook)
│   ├── config.py  logger.py  scoring.py  storage.py  drive.py
├── notebooks/
│   ├── clipgod_colab.ipynb
│   └── clipgod_kaggle.ipynb
├── scripts/colab_setup.sh  # installe faster-whisper/opencv/librosa/piper dans le notebook
├── data/                   # campaigns.txt, historical_scores.csv, model.pkl (généré)
├── inbox/  output/  cache/buffer/  logs/
└── tests/
```

---

## Installation (laptop)

Python ≥ 3.10. Seules les dépendances **légères** sont installées localement.

```bash
git clone https://github.com/moonkamel/clipgod.git
cd clipgod
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # puis remplir les clés
clipgod --help
```

Optionnel, seulement si vous activez `watch` : `pip install -e ".[watch]"`.
**N'installez pas** l'extra `[cloud]` sur le laptop : il est réservé au notebook.

### Variables d'environnement

| Variable | Utilisée par | Obtention (gratuite) |
|---|---|---|
| `YOUTUBE_API_KEY` | discover, learn | Google Cloud Console → API YouTube Data v3 → Identifiants |
| `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` | discover, watch | dev.twitch.tv/console → Register Your Application |
| `CLIPGOD_GDRIVE_CLIENT_SECRETS` | prepare, sync | Google Cloud Console → API Drive → ID client OAuth « Application de bureau » |
| `CLIPGOD_GDRIVE_TOKEN` | prepare, sync | créé automatiquement au premier login OAuth |
| `UPLOAD_POST_API_KEY` | publish | backend de publication (ou `backend: youtube_api` / `dry_run`) |
| `YOUTUBE_CLIENT_SECRETS` | publish | ID client OAuth pour l'upload YouTube |

Aucune clé n'est jamais écrite en dur ni dans `config.yaml`. Toute valeur de config peut aussi être
surchargée par variable d'environnement : `CLIPGOD__SCAN__WHISPER__MODEL=medium`.

### Connexion à Google Drive (laptop) — deux options

**Option A — Google Drive pour ordinateur (le plus simple, aucun OAuth).** Installez l'application
Google Drive, puis indiquez le dossier synchronisé dans `config.yaml` :

```yaml
drive:
  local_root: "G:/Mon Drive/ClipGod"     # Windows ; macOS : ~/Library/CloudStorage/GoogleDrive-<compte>/My Drive/ClipGod
```

**Option B — API Drive (OAuth).**
1. Google Cloud Console → créer un projet → *API et services* → activer **Google Drive API**.
2. *Écran de consentement OAuth* → type « Externe », ajoutez votre adresse comme utilisateur test.
3. *Identifiants* → *Créer* → **ID client OAuth** → « Application de bureau » → télécharger le JSON
   dans `secrets/client_secrets.json` (dossier ignoré par git).
4. Au premier `clipgod prepare` / `sync`, un navigateur s'ouvre pour l'autorisation ; le token est mis en
   cache dans `secrets/drive_token.json`.

Le scope demandé est `drive` (et non `drive.file`) : sinon le laptop ne verrait pas les clips écrits par
le notebook Colab. Pour **Kaggle**, copiez le contenu de `secrets/drive_token.json` dans un secret Kaggle
nommé `CLIPGOD_GDRIVE_TOKEN_JSON`.

---

## Workflow pas à pas

### 1. Trouver des sources — laptop

```bash
clipgod discover                       # → data/sources.json (url, title, viewers, duration, streamer, platform)
clipgod discover -p twitch -n 10       # une plateforme, 10 résultats
```

Les cibles prioritaires se déclarent dans `data/campaigns.txt` (`twitch:<login>`, `youtube:@handle`, `kick:<slug>`).

### 2. Préparer un job — laptop

```bash
clipgod prepare --source "https://www.twitch.tv/videos/123456789" --streamer kaicenat
clipgod prepare --source ./inbox/raw_20261122_2130.mp4            # fichier local (ex. sortie de watch)
```

`prepare` télécharge la VOD (yt-dlp), l'uploade dans `ClipGod/sources/` et crée `ClipGod/jobs/<job_id>.json` :

```json
{
  "job_id": "20261122-213045-a1b2c3",
  "video_id": "twitch_123456789",
  "source_drive_path": "ClipGod/sources/twitch_123456789.mp4",
  "streamer": "kaicenat",
  "platform": "twitch",
  "created_at": "2026-11-22T21:30:45Z",
  "config": { "scan": { "...": "..." }, "render": { "...": "..." } }
}
```

La commande affiche le `job_id` et le lien du notebook.

### 3. Traiter le job sur GPU gratuit — Colab

1. Ouvrez le notebook :
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moonkamel/clipgod/blob/main/notebooks/clipgod_colab.ipynb)

   ![Ouverture du notebook](docs/img/colab_01_open.png)

2. **Exécution → Modifier le type d'exécution → Accélérateur matériel : GPU T4**, puis Enregistrer.

   ![Sélection du GPU T4](docs/img/colab_02_runtime_t4.png)

3. **Exécution → Tout exécuter** (`Ctrl+F9`). À la cellule « Montage Drive », autorisez l'accès au compte
   Google **qui contient le dossier `ClipGod/`** (le même que celui utilisé par `clipgod prepare`).

   ![Autorisation Google Drive](docs/img/colab_03_mount_drive.png)

4. Le notebook clone ce dépôt, installe les dépendances (`scripts/colab_setup.sh`), prend le premier job
   non traité, puis exécute `scan` et `render`. La progression s'affiche dans la dernière cellule.

   ![Progression scan + render](docs/img/colab_04_progress.png)

5. En fin d'exécution : `ClipGod/moments/<video_id>.json`, `ClipGod/clips/<video_id>_<i>.mp4` et le
   marqueur `ClipGod/jobs_done/<job_id>.json` sont créés. Le notebook traite les jobs suivants en file s'il en reste.

   ![Résultats dans Drive](docs/img/colab_05_results.png)

> Captures : les fichiers `docs/img/colab_0*.png` sont à produire lors du premier run réel du notebook
> (étape 4 de la feuille de route) ; les emplacements sont déjà référencés ici.

**Kaggle (alternative)** : importez `notebooks/clipgod_kaggle.ipynb`, activez *Settings → Accelerator → GPU T4 x2*
et *Internet on*. Kaggle n'ayant pas de montage Drive natif, le notebook synchronise `ClipGod/` via l'API Drive
et travaille dans `/kaggle/working`.

Limites des offres gratuites à connaître : sessions Colab ~12 h max (souvent moins), déconnexion après inactivité,
quota GPU variable ; Kaggle ~30 h GPU / semaine. Grâce aux checkpoints, un job interrompu reprend où il s'était arrêté.

### 4. Récupérer les clips — laptop

```bash
clipgod sync --status        # état des jobs (en attente / terminés)
clipgod sync                 # télécharge ClipGod/clips/ et moments/ → output/
```

### 5. Valider et publier — laptop

```bash
clipgod monetize             # classement des campagnes par $/heure estimé
clipgod publish              # valide chaque clip (règles campagne) puis programme, mode test par défaut
clipgod publish --live       # publication réelle
```

Un clip refusé (durée > 30 s, pas de hook dans la 1ʳᵉ seconde, simple re-upload…) est consigné dans
`data/rejected.json` avec la raison.

### 6. Apprendre — laptop (périodique)

```bash
clipgod learn                # collecte vues 1h/6h/24h/72h, réentraîne si ≥ 20 nouveaux clips ou 7 jours, report.html
clipgod learn --retrain      # force le réentraînement
```

`data/model.pkl` est envoyé avec le job suivant : le prochain `scan` utilise automatiquement les nouveaux poids.

### Raccourci

```bash
clipgod run --source "https://www.youtube.com/watch?v=XXXX"
```

Enchaîne `discover` + `prepare`, puis affiche :

```
✔ Job 20261122-213045-a1b2c3 déposé dans ClipGod/jobs/
→ Ouvrez le notebook Colab : https://colab.research.google.com/github/moonkamel/clipgod/blob/main/notebooks/clipgod_colab.ipynb
→ Une fois terminé, lancez : clipgod sync
```

### Mode live (optionnel) — `watch`

Désactivé par défaut (`watch.disable-watch: true`) car il consomme de la bande passante en continu.

```bash
clipgod watch https://www.twitch.tv/<login> --force
```

Buffer roulant de 8 min ; quand le score de hype du chat dépasse moyenne + 2σ, les 8 min passées + 2 min suivantes
sont sauvegardées dans `inbox/raw_<timestamp>.mp4`, prêtes pour `clipgod prepare`.

---

## Comment `scan` choisit les moments

1. **Signaux par seconde** :
   - audio : loudness librosa, pics et montées brusques ;
   - texte : mots de hype, MAJUSCULES, répétitions (« no no no »), débit de parole, phrase coupée suivie d'une explosion sonore ;
   - visuel : flux optique OpenCV et coupes de scène ;
   - **most replayed** : récupéré via yt-dlp, VODs YouTube uniquement.
2. **Fenêtres glissantes** de 15 à 45 s (pas de 5 s). Chaque signal est agrégé puis normalisé par vidéo (percentiles 5-99).
3. **Score 0-100** :
   - pondérations de `scan.default_weights`, ou celles de `data/model.pkl` s'il existe ;
   - un signal absent (par ex. pas de most replayed sur Twitch) est exclu et les poids sont renormalisés ;
   - avec `model.pkl`, le score mélange l'heuristique et la prédiction du modèle. Le poids du modèle croît avec le nombre de clips d'entraînement (`model_blend`, `model_full_trust_samples`).
4. **Sélection** :
   - top 3 sans chevauchement (≥ 30 s d'écart) ;
   - bornes recalées : début sur une prise de parole, fin jamais au milieu d'un mot ;
   - durée < 30 s centrée sur le pic ;
   - sous `scan.min_score`, le moment reste un simple candidat.
5. **Hooks** : 3 accroches < 60 caractères par moment, générées par des templates locaux sans LLM. Pour l'A/B test : une citation du transcript, une accroche selon le thème (police, crash, fail…) et une selon le signal dominant.

`moments/<video_id>.json` contient pour chaque moment : `start`, `end`, `score`, `hooks`, `reason`, `signals`,
`keywords`, `transcript` et `words` (timestamps mot à mot pour les sous-titres du rendu). Les étapes coûteuses sont
checkpointées : relancer un scan, par exemple après un nouveau `model.pkl`, ne refait ni la transcription ni l'analyse vidéo.

Re-scan local sans GPU, à partir d'une transcription existante :

```bash
clipgod scan video.mp4 --transcript output/moments/<video_id>.transcript.json --streamer kaicenat
```

## Robustesse

- Logs structurés dans `logs/clipgod.log` + console.
- Checkpoints : chaque étape écrit son état ; `--resume` (défaut) relance un pipeline interrompu sans refaire le travail.
- Codes de sortie : `0` succès, `1` erreur.

## Tests

```bash
pytest
```

## Feuille de route

1. ✅ Squelette : arborescence, `pyproject.toml`, `config.yaml`, README, CLI Typer
2. ✅ `core/config.py`, `core/drive.py`, `core/logger.py`, `core/storage.py`
3. ✅ `modules/scan.py` + `core/scoring.py`
4. `notebooks/clipgod_colab.ipynb` + `scripts/colab_setup.sh`
5. `modules/render.py` + mise à jour du notebook
6. `prepare` / `sync`
7. Notebook Kaggle
8. `discover` · 9. `publish` · 10. `monetize` · 11. `learn`
12. `watch`

## Usage responsable

Respectez les conditions d'utilisation des plateformes et les règles de chaque campagne : ne clippez que des
créateurs participant à une campagne ou vous y autorisant, et transformez réellement le contenu (montage,
sous-titres, hook) plutôt que de le ré-uploader.
