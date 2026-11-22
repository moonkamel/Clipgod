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

> 📘 **Guide pas à pas, du laptop vierge au premier clip publié : [docs/INSTALLATION.md](docs/INSTALLATION.md)**

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
clipgod discover                       # → data/sources.json + tableau des meilleures sources
clipgod discover -p twitch -n 10       # une plateforme, 10 résultats
clipgod discover --no-cache            # ignore le cache de 5 min
clipgod run                            # discover + prepare de la meilleure VOD, en une commande
```

| Plateforme | Accès (gratuit) | Ce qui est examiné |
|---|---|---|
| YouTube | Data API v3 (`YOUTUBE_API_KEY`) | 1 recherche de lives et 1 de VODs longues des dernières 24 h (100 unités chacune), plus les dernières vidéos des chaînes de `campaigns.txt` (1 unité). |
| Twitch | Helix (`TWITCH_CLIENT_ID` / `SECRET`) | Lives de la catégorie *Grand Theft Auto VI*, top VODs du jour de la catégorie, chaînes de `campaigns.txt`. |
| Kick | Lecture légère des endpoints JSON publics | Lives et VODs des chaînes de `campaigns.txt` / `discover.kick.channels`. Ces endpoints ne sont pas officiels : si Kick les change ou les bloque, la plateforme est ignorée avec un avertissement. |

**Filtres** :
- live en cours depuis au moins 45 min, ou VOD de moins de 24 h et de plus de 45 min ;
- titre ou catégorie GTA 6 (« GTA V » est exclu) ;
- `min_viewers`, sauf pour les chaînes de campagne.

**Classement** : les chaînes de `data/campaigns.txt` d'abord, puis le ratio **vues/heure** (VOD : vues ÷ heures
depuis la publication ; live : spectateurs simultanés). Une plateforme sans clé ou en panne est ignorée sans
bloquer les autres.

**Quota YouTube** : il est partagé avec `publish` et suivi dans `data/state.json`, avec une remise à zéro à minuit
heure du Pacifique. `discover` est plafonné à `discover.youtube.daily_budget` (1 500 unités par jour) pour laisser
de la place aux uploads (1 600 unités chacun).

`data/campaigns.txt` :

```
twitch:kaicenat gta6-launch     # plateforme:identifiant [nom de campagne]
youtube:@caseoh
kick:xqc
```

### 2. Préparer un job — laptop

```bash
clipgod prepare --source "https://www.twitch.tv/videos/123456789" --streamer kaicenat
clipgod prepare --source ./inbox/raw_20261122_2130.mp4            # fichier local (ex. sortie de watch)
```

`prepare` crée `ClipGod/jobs/<job_id>.json`. La façon dont la vidéo arrive sur le GPU dépend de
`prepare.download`, ou de l'option `--download` :

| Mode | Comportement |
|---|---|
| `auto` (défaut) | **YouTube** : téléchargé sur le laptop (yt-dlp), puis uploadé dans `ClipGod/sources/`, car YouTube bloque souvent les IP Colab. **Twitch / Kick** : téléchargé directement par le notebook, ce qui épargne la connexion du laptop. |
| `local` | Toujours télécharger sur le laptop, puis uploader. ffmpeg est requis sur le laptop. |
| `cloud` | Le notebook télécharge toujours lui-même. |

Un fichier local, par exemple une sortie de `watch`, est toujours uploadé. Le graphe *most replayed* de YouTube
est récupéré sur le laptop et embarqué dans le job. `data/model.pkl` est envoyé dans `ClipGod/model.pkl`
s'il a changé. Un 2ᵉ `prepare` de la même vidéo est refusé, sauf avec `--force`.

```json
{
  "job_id": "20261122-213045-a1b2c3",
  "video_id": "twitch_2345678",
  "platform": "twitch",
  "source_url": "https://www.twitch.tv/videos/2345678",
  "source_drive_path": null,
  "download": "cloud",
  "streamer": "kaicenat",
  "heatmap": null,
  "options": { "voiceover": false },
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

4. Le notebook clone ce dépôt, installe les dépendances (`scripts/colab_setup.sh`), prend les jobs
   en attente du plus ancien au plus récent, puis exécute `scan` et `render`. La progression s'affiche dans la
   cellule 7️⃣, et la cellule 8️⃣ montre les moments détectés avec leurs hooks.

   Paramètres (cellule ⚙️) :
   - `BRANCH` : branche à cloner. Mettez celle où se trouve le code tant qu'il n'est pas fusionné dans `main`.
   - `STAGES` : `scan`, `scan+render` ou `render`.
   - `MAX_JOBS`, `ONLY_JOB` : limiter le traitement à quelques jobs ou à un job précis.
   - `WHISPER_MODEL` : modèle forcé (`auto` par défaut).

   **Repo privé** : ajoutez un secret Colab `GITHUB_TOKEN` (icône 🔑, jeton en lecture seule).

   ![Progression scan + render](docs/img/colab_04_progress.png)

5. En fin d'exécution : `ClipGod/moments/<video_id>.json`, `ClipGod/clips/<video_id>_<i>.mp4` et le
   marqueur `ClipGod/jobs_done/<job_id>.json` sont créés. Le notebook traite les jobs suivants en file s'il en reste.

   ![Résultats dans Drive](docs/img/colab_05_results.png)

**Comportement du worker** (`core/worker.py`) :
- **Réservation** : chaque job est réservé via `ClipGod/jobs/<job_id>.lock` (bail de 3 h). Deux sessions,
  par exemple Colab et Kaggle, ne traitent donc pas le même job.
- **Échec** : le job reste en file et sera retenté au run suivant. Au 2ᵉ échec, ou si le job est invalide,
  il est marqué `failed` dans `jobs_done/` avec l'erreur.
- **Checkpoints** : ceux du scan sont écrits sur Drive (`ClipGod/moments/.work/`). Après une déconnexion,
  relancez *Tout exécuter* : la transcription déjà faite n'est pas refaite.
- **Poids appris** : si `ClipGod/model.pkl` existe (déposé par le laptop après `clipgod learn`), il est
  utilisé automatiquement.

> Captures : les fichiers `docs/img/colab_0*.png` sont à produire lors du premier run réel du notebook
> (étape 4 de la feuille de route) ; les emplacements sont déjà référencés ici.

**Kaggle (alternative, ≈ 30 h de GPU par semaine)**

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/moonkamel/clipgod/blob/main/notebooks/clipgod_kaggle.ipynb)

1. Importez `notebooks/clipgod_kaggle.ipynb`, ou utilisez le badge (repo public).
2. Panneau de droite :
   - *Accelerator* : **GPU T4 x2** ;
   - *Internet* : **On**. Il faut un compte Kaggle vérifié par téléphone.
3. *Add-ons › Secrets* : créez `CLIPGOD_GDRIVE_TOKEN_JSON` avec le **contenu** de `secrets/drive_token.json` du laptop.
   Ce fichier est créé au premier `clipgod prepare` / `sync` en mode API. Ajoutez aussi `GITHUB_TOKEN` si le repo
   est privé.
4. **Run All**. Pour une longue VOD sans garder l'onglet ouvert : *Save Version › Save & Run All*, qui
   s'exécute en arrière-plan jusqu'à 12 h.

Kaggle n'a pas de Drive monté : le notebook passe par l'API Drive. Les checkpoints locaux disparaissent avec la
session, mais une transcription déjà envoyée dans `ClipGod/moments/` est réutilisée à la relance. Le token n'est
jamais écrit dans les sorties du notebook, et l'URL du repo cloné est nettoyée.

Limites des offres gratuites à connaître : sessions Colab ~12 h max (souvent moins), déconnexion après inactivité,
quota GPU variable ; Kaggle ~30 h GPU / semaine. Grâce aux checkpoints, un job interrompu reprend où il s'était arrêté.

### 4. Récupérer les clips — laptop

```bash
clipgod sync --status        # état de la file : queued / running / retry / done / failed / synced
clipgod sync                 # télécharge clips (.mp4 + .json) et moments → output/, met à jour data/state.json
clipgod sync --job-id <id>   # un seul job
```

Après synchronisation, la vidéo source est retirée de Drive (`sync.cleanup_sources`, quota gratuit de 15 Go).
Un job en échec est affiché avec son erreur. La commande est idempotente : relancer `sync` ne retélécharge rien.

### 5. Valider et publier — laptop

```bash
clipgod publish --plan          # montre le plan (validation, plateformes, créneaux) sans rien envoyer
clipgod publish                 # mode test (défaut) : upload YouTube en PRIVÉ, rien de public
clipgod publish --live          # programme sur les créneaux de forte audience (max 6/jour)
clipgod publish --live --now    # publie immédiatement (toujours plafonné à 6/jour)
```

1. **Validation** (module `monetize`) :
   - durée < 30 s, hook dans la 1ʳᵉ seconde, format vertical ;
   - vraie transformation : recadrage, plus au moins 2 éléments parmi sous-titres, hook, facecam et voix off.

   Un clip refusé va dans `data/rejected.json` avec ses raisons.
2. **A/B test** : les variantes de hook d'un même moment partent sur des **créneaux différents** et tournent entre
   les plateformes. Les meilleurs moments passent en premier, et deux variantes d'un même moment ne se suivent jamais.
3. **Créneaux** : `publish.slots`, en heure de Paris : matin EU (08:30, 11:00) et soirée US (00:30 à 03:30, soit 18h30-21h30 EST).
4. **Plateformes** (`publish.routes`) :

| Plateforme | Backend | Prérequis |
|---|---|---|
| YouTube Shorts | API YouTube officielle (gratuite), programmation native `publishAt` | `YOUTUBE_CLIENT_SECRETS` : ID client OAuth « Application de bureau », API YouTube Data v3 activée. Navigateur ouvert au 1er lancement. |
| TikTok, Instagram Reels | [Upload-Post](https://upload-post.com) (optionnel) | `UPLOAD_POST_API_KEY` et `UPLOAD_POST_USER`. Sans clé, ces plateformes sont simplement ignorées. |

> ⚠️ Tant que votre projet Google Cloud n'a pas passé l'**audit de l'API YouTube**, Google force en privé toutes
> les vidéos envoyées par l'API. Le mode test fonctionne donc immédiatement. La publication publique sur YouTube
> demande l'audit (gratuit, formulaire Google), ou passe par Upload-Post.

Le quota YouTube est vérifié avant chaque upload (1 600 unités sur 10 000 par jour, partagé avec `discover`). S'il
est épuisé, YouTube est mis en pause pour le run et les autres plateformes continuent. Chaque post est enregistré
dans `data/state.json` (plateforme, créneau, URL) : un clip n'est jamais publié deux fois sur la même plateforme.

### Choisir ses campagnes — `monetize`

```bash
clipgod monetize --refresh        # scrape Whop Content Rewards (cache 30 min) et classe les campagnes
clipgod monetize --all            # inclut les campagnes exclues (CPM trop bas, budget épuisé, inactive)
clipgod monetize --validate output/clips/<clip>.mp4 --campaign kai-gta6
clipgod prepare --source <url> --campaign kai-gta6     # rattache les clips de cette VOD à la campagne
```

- **Sources** : la page publique Content Rewards, plus une page par campagne suivie (lignes `whop:<slug>` de
  `data/campaigns.txt`). Le scraping est léger : 1 page par seconde maximum, avec cache.
  - L'extraction lit d'abord les données JSON embarquées par la page, puis le texte visible (« $1.50 per 1K
    views », « max 60 seconds », `#hashtag`, `@mention`…).
  - **Le site de Whop évolue** : si l'extraction rate une campagne, ajoutez-la dans **`data/whop_campaigns.yaml`**.
    Cette saisie manuelle prime toujours sur le scraping.
- **Classement en $/heure de VOD** = vues attendues par clip × CPM ÷ 1000 × clips par heure de VOD, plafonné par
  le budget restant.
  - Vues attendues : la médiane de `historical_scores.csv` pour cette campagne, sinon toutes campagnes
    confondues, sinon `prior_views_per_clip`.
  - Clips par heure de VOD : mesuré sur vos jobs, sinon `clips_per_vod_hour`.
- **Règles par campagne**, appliquées par `publish` :
  - durées min/max, sous-titres exigés ;
  - plateformes autorisées (les autres sont sautées) ;
  - hashtags et mentions obligatoires, ajoutés automatiquement au titre et à la description.

  Un clip sans campagne explicite est rattaché à celle qui cible son streamer (`streamers:`).

### 6. Apprendre — laptop (périodique)

```bash
clipgod learn                # mesure les vues, met à jour historical_scores.csv, réentraîne si besoin, report.html
clipgod learn --retrain      # force le réentraînement
```

Lancez-le idéalement **1 h, 6 h, 24 h et 72 h après vos publications**. Les horaires peuvent être irréguliers :
chaque exécution ajoute une mesure, et les vues aux checkpoints sont **interpolées** entre deux mesures, jamais
extrapolées.

| Source des vues | Comment |
|---|---|
| YouTube | API officielle (1 unité de quota pour 50 vidéos). Les vidéos privées du mode test sont visibles via le token OAuth de `publish`. |
| TikTok / Instagram | `publish.upload_post.analytics_endpoint` si votre offre Upload-Post l'expose, sinon saisie dans `data/manual_views.csv` (`url,views,measured_at`). |

- **Historique** : `data/historical_scores.csv`, une ligne par post (`clip_id, signals, hook, platform,
  views_1h, views_6h, views_24h, views_72h`, plus la campagne, la variante, l'URL…).
- **Modèle** : un GradientBoostingRegressor qui prédit, à partir des signaux du moment, `log(vues)` moins la
  médiane de la plateforme. Il est réentraîné tous les 20 nouveaux clips ou chaque semaine (à partir de 10 clips
  mesurés), puis écrit dans `data/model.pkl` et **envoyé sur Drive** : le prochain `scan`, sur Colab, Kaggle ou en
  local, l'utilise automatiquement. Son poids dans le score augmente avec le nombre de clips d'entraînement.
- **`report.html`** (autonome, sans JavaScript) contient :
  - indicateurs clés, recommandations ;
  - top et flop des clips ;
  - hooks gagnants aux tests A/B, styles de hook ;
  - meilleurs créneaux et plateformes ;
  - campagnes les plus rentables (gains estimés = vues × CPM) ;
  - importance de chaque signal dans le modèle.

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

## Comment `render` fabrique les clips

| Élément | Détail |
|---|---|
| Layout | YuNet (OpenCV) analyse le moment à 2 fps. Un visage **stable et petit** est considéré comme la facecam du streamer : gameplay en haut (70 %), facecam en bas (30 %) sur fond flouté, avec une bande de flou entre les deux. Sinon : recadrage 9:16 plein écran, centré, qui suit en douceur un visage éventuel. |
| Facecam forcée | `render.layout.facecam_regions: {kaicenat: [x, y, w, h]}` (coordonnées 0..1) quand la détection se trompe. |
| Sous-titres | Mot à mot depuis les timestamps whisper : blocs de 3 mots maximum, mot courant en jaune, police Montserrat ExtraBold avec contour noir, à 64 % de la hauteur. |
| Hook | Boîte blanche animée (*pop*) pendant les 2 premières secondes. **Une variante de clip par hook** pour l'A/B test : `<id>_<i>.mp4`, `<id>_<i>_h1.mp4`… |
| Audio | `loudnorm` en deux passes : −14 LUFS mesurés. Voix off Piper en option (lit le hook et baisse le son d'origine pendant la lecture). |
| Overlay | `@streamer` en bas pendant toute la durée. |
| Encodage | H.264 1080×1920 à 30 fps : NVENC sur T4, libx264 en repli automatique. AAC 48 kHz, `+faststart`. |

Chaque clip a un fichier `.json` voisin : hook, timing, score, signaux, layout et `transformations` appliquées.
`monetize` s'en servira pour prouver que le clip n'est pas un simple re-upload. Un clip déjà rendu avec la
même config est réutilisé, ce qui permet de reprendre après une interruption.

Re-rendu local d'un moment, pour du debug :

```bash
clipgod render source.mp4 -m output/moments/<video_id>.json --only 1 --layout center --streamer kaicenat
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
4. ✅ `notebooks/clipgod_colab.ipynb` + `scripts/colab_setup.sh` + `core/worker.py`
5. ✅ `modules/render.py` + mise à jour du notebook
6. ✅ `prepare` / `sync`
7. ✅ Notebook Kaggle
8. ✅ `discover` · 9. ✅ `publish` · 10. ✅ `monetize` · 11. ✅ `learn`
12. `watch`

## Usage responsable

Respectez les conditions d'utilisation des plateformes et les règles de chaque campagne : ne clippez que des
créateurs participant à une campagne ou vous y autorisant, et transformez réellement le contenu (montage,
sous-titres, hook) plutôt que de le ré-uploader.
