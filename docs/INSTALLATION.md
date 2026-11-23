# Guide d'installation de ClipGod

Ce guide part de zéro : un laptop (Windows, macOS ou Linux), un compte Google et une connexion Internet.
Comptez **environ 1 h** la première fois, dont la moitié pour créer les clés d'API.

> Tout est gratuit. Seul Upload-Post (publication TikTok / Instagram) est optionnel et peut être payant au-delà
> de son offre gratuite.

## Sommaire

1. [Ce qu'il vous faut](#1-ce-quil-vous-faut)
2. [Installer les outils du laptop](#2-installer-les-outils-du-laptop)
3. [Récupérer ClipGod](#3-récupérer-clipgod)
4. [Installer ClipGod](#4-installer-clipgod)
5. [Créer les clés Google (Drive + YouTube)](#5-créer-les-clés-google-drive--youtube)
6. [Créer les clés Twitch](#6-créer-les-clés-twitch)
7. [Remplir le fichier `.env`](#7-remplir-le-fichier-env)
8. [Connecter Google Drive](#8-connecter-google-drive)
9. [Préparer le notebook GPU (Colab)](#9-préparer-le-notebook-gpu-colab)
10. [Option : Kaggle](#10-option--kaggle)
11. [Option : TikTok et Instagram](#11-option--tiktok-et-instagram)
12. [Cibler vos chaînes et campagnes](#12-cibler-vos-chaînes-et-campagnes)
13. [Premier essai de bout en bout](#13-premier-essai-de-bout-en-bout)
14. [Automatiser `learn`](#14-automatiser-learn)
15. [Top 5 GTA 6 du jour, chaque matin](#15-top-5-gta-6-du-jour-chaque-matin)
16. [Dépannage](#16-dépannage)

---

## 1. Ce qu'il vous faut

| Élément | Pourquoi | Obligatoire |
|---|---|---|
| Un compte **Google** | Google Drive (lien laptop ↔ GPU), Colab, YouTube | Oui |
| Un compte **Twitch** | Clés de l'API Twitch pour `discover` | Recommandé |
| Une chaîne **YouTube** | Publication des Shorts | Pour `publish` |
| Un compte **GitHub** | Seulement si le dépôt ClipGod est privé | Selon le dépôt |
| Un compte **Kaggle** vérifié par téléphone | GPU de secours (30 h/semaine) | Non |
| Un compte **Upload-Post** | Publier sur TikTok / Instagram | Non |

Le laptop n'a **pas besoin de GPU** : il ne fait qu'orchestrer. La transcription et le montage tournent
sur le GPU gratuit de Google Colab.

Espace disque sur le laptop : ~1 Go pour ClipGod et ses dépendances, plus la place des clips (~5 Mo par clip).

---

## 2. Installer les outils du laptop

Il faut **Python 3.10 ou plus récent**, **Git** et **ffmpeg**.

### Windows

Ouvrez **PowerShell** (menu Démarrer › « PowerShell ») et lancez :

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
winget install -e --id Gyan.FFmpeg
```

Fermez puis rouvrez PowerShell pour que les nouvelles commandes soient trouvées.

> Sans `winget` : installez Python depuis [python.org](https://www.python.org/downloads/) en cochant
> **« Add python.exe to PATH »** sur le premier écran, Git depuis [git-scm.com](https://git-scm.com/download/win),
> et ffmpeg depuis [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (ajoutez son dossier `bin` au PATH).

### macOS

Avec [Homebrew](https://brew.sh) :

```bash
brew install python@3.12 git ffmpeg
```

### Linux (Debian / Ubuntu)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git ffmpeg
```

### Vérification

```bash
python --version      # Windows (ou python3 --version sur macOS / Linux) : 3.10 ou plus
git --version
ffmpeg -version
```

> ffmpeg sert au laptop pour télécharger les VODs YouTube (assemblage vidéo + audio) et, plus tard, pour `watch`.
> Sans lui, utilisez `--download cloud` : c'est alors le notebook qui télécharge.

---

## 3. Récupérer ClipGod

Placez-vous dans le dossier où vous voulez installer ClipGod (par exemple vos Documents), puis :

```bash
git clone https://github.com/moonkamel/clipgod.git
cd clipgod
```

**Tant que le code n'est pas fusionné dans `main`**, prenez la branche de développement :

```bash
git checkout ccr-80e6fd10-jyuvxf
```

Le dépôt est privé ? Git vous demandera un identifiant et un mot de passe : utilisez votre nom d'utilisateur GitHub
et un **jeton d'accès** (GitHub › Settings › Developer settings › Personal access tokens › *Fine-grained*,
accès en lecture au contenu du dépôt), pas votre mot de passe.

---

## 4. Installer ClipGod

Créez un environnement Python isolé, puis installez ClipGod dedans.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

> Si PowerShell refuse d'activer l'environnement (« l'exécution de scripts est désactivée ») :
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis relancez la commande d'activation.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Vérifiez :

```bash
clipgod --help
```

`clipgod --help` doit lister les commandes.

> Ne lancez pas `pytest` sur le laptop : une partie des tests a besoin des dépendances GPU (faster-whisper,
> OpenCV…), qui ne sont installées que sur Colab / Kaggle. Les tests tournent automatiquement sur GitHub.

> **À chaque nouvelle session de terminal**, réactivez l'environnement (`.venv\Scripts\Activate.ps1` ou
> `source .venv/bin/activate`) avant d'utiliser `clipgod`.
>
> N'installez **pas** l'extra `[cloud]` sur le laptop : il contient faster-whisper et OpenCV, réservés au notebook GPU.

---

## 5. Créer les clés Google (Drive + YouTube)

Tout se fait dans la [Google Cloud Console](https://console.cloud.google.com/), avec le compte Google qui
possède votre Drive et votre chaîne YouTube.

### 5.1 Créer un projet

En haut de la page : sélecteur de projet › **Nouveau projet** › nom `clipgod` › **Créer**, puis sélectionnez-le.

### 5.2 Activer les API

Menu › **API et services** › **Bibliothèque**, puis activez :

- **Google Drive API**
- **YouTube Data API v3**

### 5.3 Écran de consentement OAuth

Menu › **API et services** › **Écran de consentement OAuth** (ou « Google Auth Platform ») :

1. Type d'utilisateur : **Externe**.
2. Nom de l'application : `ClipGod` ; adresse d'assistance et adresse du développeur : la vôtre.
3. **Utilisateurs test** : ajoutez **votre adresse Gmail**. Sans cette étape, Google affichera « Accès bloqué ».

> En mode « Test », Google fait expirer l'autorisation au bout de **7 jours** : il faudra alors vous reconnecter
> (voir [Dépannage](#15-dépannage)). Pour l'éviter, passez l'application en **« En production »** sur ce même
> écran. Pour un usage personnel, aucune validation n'est nécessaire : Google affiche seulement un avertissement
> « application non validée » que vous pouvez passer (*Paramètres avancés › Accéder à ClipGod*).

### 5.4 Identifiant OAuth (Drive et upload YouTube)

Menu › **API et services** › **Identifiants** › **Créer des identifiants** › **ID client OAuth** :

- Type d'application : **Application de bureau** ; nom : `ClipGod laptop` › **Créer**.
- Cliquez **Télécharger le JSON**.
- Dans le dossier `clipgod`, créez un dossier `secrets` et enregistrez-y ce fichier sous le nom
  **`client_secrets.json`**.

Le dossier `secrets/` est ignoré par Git : vos clés ne partiront jamais sur GitHub.

### 5.5 Clé d'API YouTube (recherche de vidéos)

Toujours dans **Identifiants** › **Créer des identifiants** › **Clé API**. Copiez-la, puis cliquez
**Modifier la clé** › *Restrictions relatives aux API* › **YouTube Data API v3** uniquement › Enregistrer.

---

## 6. Créer les clés Twitch

1. Connectez-vous sur la [console développeur Twitch](https://dev.twitch.tv/console) (activez la double
   authentification sur votre compte Twitch si on vous le demande).
2. **Applications** › **Enregistrer votre application** :
   - Nom : `clipgod-<votre pseudo>` (le nom doit être unique sur Twitch) ;
   - URL de redirection OAuth : `http://localhost` ;
   - Catégorie : *Application Integration* ;
   - Type de client : **Confidentiel**.
3. Ouvrez l'application créée : copiez l'**ID client**, puis **Nouveau secret** et copiez le secret
   (il n'est affiché qu'une fois).

---

## 7. Remplir le fichier `.env`

Copiez le modèle :

```bash
cp .env.example .env          # Windows : copy .env.example .env
```

Ouvrez `.env` avec un éditeur de texte (Bloc-notes, VS Code…) et remplissez :

```ini
YOUTUBE_API_KEY=AIza...                 # étape 5.5
TWITCH_CLIENT_ID=abc123...              # étape 6
TWITCH_CLIENT_SECRET=xyz789...          # étape 6
CLIPGOD_GDRIVE_CLIENT_SECRETS=./secrets/client_secrets.json
CLIPGOD_GDRIVE_TOKEN=./secrets/drive_token.json
YOUTUBE_CLIENT_SECRETS=./secrets/client_secrets.json
```

Le même `client_secrets.json` sert à Drive et à YouTube. Les fichiers `*_token.json` seront créés
automatiquement à la première connexion.

> Ne partagez jamais `.env` ni le dossier `secrets/`. Ils sont exclus de Git par `.gitignore`.

---

## 8. Connecter Google Drive

Le laptop et le notebook GPU s'échangent les fichiers via le dossier **`ClipGod/`** de votre Google Drive.
Deux façons de le relier au laptop.

### Option A : Google Drive pour ordinateur (le plus simple)

Si l'application [Google Drive pour ordinateur](https://www.google.com/drive/download/) est installée et
synchronise votre Drive, ouvrez `config.yaml` et indiquez le dossier :

```yaml
drive:
  local_root: "G:/Mon Drive/ClipGod"     # Windows (vérifiez la lettre de lecteur)
  # macOS : "~/Library/CloudStorage/GoogleDrive-<votre adresse>/My Drive/ClipGod"
```

Pas d'autorisation à donner : ClipGod lit et écrit directement dans ce dossier.

### Option B : API Drive (sans application à installer)

Laissez `local_root: null` et lancez :

```bash
clipgod sync --status
```

Un navigateur s'ouvre : choisissez votre compte Google, acceptez l'accès à Drive. Le terminal affiche ensuite
« Aucun job sur Drive » : la connexion fonctionne, et le token est enregistré dans `secrets/drive_token.json`.

Dans les deux cas, ClipGod crée le dossier `ClipGod/` (avec `jobs/`, `sources/`, `moments/`, `clips/`,
`jobs_done/`) à la première utilisation.

---

## 9. Préparer le notebook GPU (Colab)

1. Ouvrez [Google Colab](https://colab.research.google.com) avec **le même compte Google**.
2. **Fichier › Ouvrir un notebook › GitHub** : collez `moonkamel/clipgod`, choisissez la branche
   (`ccr-80e6fd10-jyuvxf` tant que le code n'est pas dans `main`) et ouvrez `notebooks/clipgod_colab.ipynb`.
   Une fois le code fusionné dans `main`, le badge *Open in Colab* du README suffit.
3. **Exécution › Modifier le type d'exécution** › Accélérateur matériel : **GPU T4** › Enregistrer.
4. Dans la cellule **⚙️ Paramètres**, mettez `BRANCH` sur la même branche que ci-dessus.
5. **Dépôt privé** : icône 🔑 (Secrets) à gauche › ajoutez `GITHUB_TOKEN` avec un jeton GitHub en lecture
   seule, et activez « Accès au notebook ».

Pas besoin de lancer le notebook maintenant : vous le ferez à l'étape 13, quand un job l'attendra.

---

## 10. Option : Kaggle

Utile quand votre quota Colab est épuisé.

1. Sur [kaggle.com](https://www.kaggle.com), vérifiez votre compte par téléphone (*Settings › Phone verification*).
2. **Create › New Notebook** › *File › Import Notebook* › importez `notebooks/clipgod_kaggle.ipynb`.
3. Panneau de droite : *Session options* › Accelerator **GPU T4 x2**, Internet **On**.
4. *Add-ons › Secrets* :
   - `CLIPGOD_GDRIVE_TOKEN_JSON` : collez **tout le contenu** du fichier `secrets/drive_token.json` du laptop.
     Il n'existe qu'après l'option B de l'étape 8 (lancez `clipgod sync --status` une fois même si vous utilisez
     l'option A) ;
   - `GITHUB_TOKEN` si le dépôt est privé.
5. Dans la cellule de paramètres, ajustez `BRANCH`.

---

## 11. Option : TikTok et Instagram

### Gratuit : publication manuelle assistée (`clipgod tiktok`)

Aucun compte ni clé d'API : vous déposez vous-même les clips dans
[TikTok Studio](https://www.tiktok.com/tiktokstudio/upload), qui programme jusqu'à 10 jours à l'avance.

```bash
clipgod sync              # récupère les clips depuis Drive
clipgod tiktok --plan     # aperçu des prochains posts et de leurs légendes
clipgod tiktok            # séance de publication (3 clips par défaut, -n 5 pour 5)
clipgod tiktok --choose   # liste tous les clips : « v3 » pour regarder le 3, puis « 1,3,5-7 » pour choisir
clipgod tiktok --export   # dossier output/a_publier : 1 vidéo par moment, nommée d'après son hook + legendes.txt
```

Pour chaque clip, ClipGod :

1. copie la légende (hook, crédit du streamer, hashtags et mentions de la campagne) dans le presse-papiers ;
2. ouvre l'explorateur sur le clip, déjà sélectionné : glissez-le dans TikTok Studio ;
3. indique le créneau de forte audience à saisir dans « Programmer » ;
4. attend votre confirmation. Collez le lien de la vidéo TikTok si vous l'avez : il servira à `learn`.

Un clip confirmé n'est jamais reproposé, et une seule variante de hook par moment part sur TikTok.
Réglages : `publish.manual` dans `config.yaml` (`max_per_day: 3` par défaut, hashtags par plateforme).
Pour Instagram : `clipgod tiktok --platform instagram_reels`.

### Payant : envoi automatique (Upload-Post)

La connexion TikTok n'est pas incluse dans l'offre gratuite d'Upload-Post.
Sans cette étape, `publish` utilise uniquement YouTube Shorts.

1. Créez un compte sur [upload-post.com](https://upload-post.com), connectez-y vos comptes TikTok / Instagram,
   et notez le nom du **profil** créé.
2. Récupérez votre clé d'API dans leur tableau de bord.
3. Dans `.env` :

   ```ini
   UPLOAD_POST_API_KEY=...
   UPLOAD_POST_USER=nom_du_profil
   ```

4. Comparez l'URL et les champs d'envoi avec leur documentation, et ajustez si besoin `publish.upload_post`
   dans `config.yaml`.

---

## 12. Cibler vos chaînes et campagnes

- **`data/campaigns.txt`** : les chaînes à surveiller en priorité et les campagnes Whop suivies, une par ligne :

  ```
  twitch:nom_de_la_chaine  nom-de-campagne
  youtube:@handle
  kick:slug
  whop:slug-de-la-campagne
  ```

- **`data/whop_campaigns.yaml`** : collez-y les campagnes Whop qui vous intéressent (CPM, budget, règles,
  hashtags obligatoires). Le fichier contient un exemple commenté. Ces informations priment sur le scraping
  automatique, qui peut rater des campagnes si Whop change son site.

Vérifiez avec :

```bash
clipgod monetize --refresh
```

---

## 13. Premier essai de bout en bout

Chaque étape doit réussir avant la suivante.

```bash
# 1. Trouver des sources (YouTube + Twitch)
clipgod discover

# 2. Envoyer une VOD au notebook (prenez une URL affichée par discover, ou --source chemin/vers/video.mp4)
clipgod prepare --source https://www.twitch.tv/videos/XXXXXXXXX
```

3. Dans Colab : **Exécution › Tout exécuter**. Autorisez l'accès à Drive quand on vous le demande.
   Comptez ~10 à 20 min pour une VOD de 2 h (la première fois, le modèle whisper est téléchargé).
   La cellule 8️⃣ montre les moments détectés et le premier clip.

```bash
# 4. Récupérer les clips sur le laptop
clipgod sync

# 5. Voir le plan de publication, puis publier en mode test (YouTube, en privé)
clipgod publish --plan
clipgod publish

# 6. Plus tard (1 h, 6 h, 24 h après) : mesurer les vues et générer le rapport
clipgod learn
```

Au premier `publish`, un navigateur s'ouvre pour autoriser l'envoi sur votre chaîne YouTube.
Les vidéos apparaissent en **privé** dans YouTube Studio.

Pour publier réellement : `clipgod publish --live`. Tant que votre projet Google Cloud n'a pas passé l'audit
de l'API YouTube, Google garde ces vidéos en privé (voir [Dépannage](#15-dépannage)).

Raccourci pour les étapes 1-2 : `clipgod run` (meilleure VOD trouvée) ou `clipgod run --source <url>`.

---

## 14. Automatiser `learn`

`learn` doit tourner régulièrement pour mesurer les vues à 1 h, 6 h, 24 h et 72 h. Une exécution toutes les
heures suffit.

**Windows (Planificateur de tâches)** : créez un fichier `learn.bat` dans le dossier `clipgod` :

```bat
@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
clipgod learn >> logs\learn_task.log 2>&1
```

puis : Planificateur de tâches › *Créer une tâche de base* › Déclencheur **Quotidien**, répéter **toutes les
1 heure** › Action : démarrer `learn.bat`.

**macOS / Linux (cron)** : `crontab -e` puis ajoutez :

```cron
7 * * * * cd /chemin/vers/clipgod && .venv/bin/clipgod learn >> logs/learn_task.log 2>&1
```

---

## 15. Top 5 GTA 6 du jour, chaque matin

Le cycle quotidien tient en une tâche planifiée et un clic sur Colab :

| Moment | Où | Ce qui se passe |
|---|---|---|
| Le matin (tâche planifiée) | PC | `git pull` → `clipgod sync` (clips de la veille) → `clipgod bestof --viral` (le Top 5 du jour) → `clipgod viral` (les moments des Shorts les plus vus) → Colab s'ouvre |
| Juste après | Colab | **Exécution › Tout exécuter**, onglet ouvert jusqu'à la fin (~10 min pour 5 extraits) |
| Le lendemain matin | PC | la tâche récupère ces clips et monte le Top 5 suivant |

**Installer la tâche** (une seule fois, dans PowerShell, depuis le dossier `clipgod`) :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -At 09:15
```

(`-Mode trending` pour l'ancien fonctionnement : VODs entières du jour, analysées de bout en bout.)

Si le PC est éteint à l'heure prévue, la tâche part au démarrage suivant. Pour la lancer tout de suite :
`Start-ScheduledTask -TaskName "ClipGod quotidien"`. Journal : `logs\daily_<date>.txt`.
Pour la supprimer : `Unregister-ScheduledTask -TaskName "ClipGod quotidien" -Confirm:$false`.

**Mode viral (par défaut)** : le Top 5 reprend les moments des **Shorts GTA 6 les plus vus** des dernières 24 h,
classés par vues. Pour chaque Short, `clipgod viral` retrouve la VOD d'origine (lien de la description, même chaîne,
@mention), écoute sa piste audio pour localiser **la seconde exacte** du moment (empreinte audio), puis télécharge
seulement les ~30-50 s utiles et les dépose sur Drive. Colab les sous-titre et les monte en 1-2 min chacun.
Le Short lui-même n'est jamais réutilisé dans le montage.

```powershell
clipgod viral --plan     # localise les moments sans rien envoyer (pour voir)
clipgod viral            # localise + envoie les extraits sur Drive -> Colab : Tout exécuter
clipgod sync
clipgod bestof --viral   # Top 5 classé par vues des Shorts -> output\bestof\
```

Un Short dont la VOD d'origine est introuvable (pas de lien, VOD supprimée, Short accéléré…) est ignoré :
`clipgod viral` en examine jusqu'à 25 pour en garder 5. Réglages : section `viral` de `config.yaml`.

**Veille : les Shorts gameplay les plus vus**

```powershell
clipgod topshorts               # top 10 des 48 h, gameplay uniquement -> top10_shorts_gta6.txt
clipgod topshorts --hours 24    # sur 24 h ; --all pour enlever le filtre ; -n 20 pour 20 Shorts
```

Le filtre lit le titre, les tags et la description : un Short de réaction, de comparaison (« GTA 5 vs GTA 6 »),
d'actualité, de théorie ou de stream est écarté ; un Short qui parle de gameplay (crash, police, mission, physique,
secret…) est gardé. Les Shorts écartés sont listés en bas du fichier pour vérifier qu'aucun n'a été oublié.

**Veille TikTok** (une fois : `pip install playwright`, aucun navigateur à télécharger) :

```powershell
clipgod toptiktok               # top 10 gameplay des 48 h -> top10_tiktok_gta6.txt
clipgod toptiktok --hours 24 --scrolls 15   # sur 24 h, en lisant plus de vidéos
```

TikTok n'a pas d'API gratuite : ClipGod ouvre les pages #gta6, #gtavi et #gta6gameplay dans **ton Chrome** (ou
Edge), fait défiler, et lit les données que la page télécharge pour s'afficher : lien, **vues**, description et
**langue détectée par TikTok** (à défaut, devinée d'après la description). Une vidéo sans vues ni langue dans la
liste est relue sur sa propre page (60 au plus, ~2 s chacune). Par défaut, **seules les vidéos en anglais** sont
gardées : `--lang en,fr` pour ajouter le français, `--lang all` pour toutes. La date de publication est déduite
de l'identifiant de chaque vidéo. Ne ferme pas la fenêtre pendant la lecture (~1 min). Si TikTok affiche une
vérification, résous-la une fois : les cookies sont gardés dans `cache\tiktok_profile`. Seules les vidéos chargées
sur ces pages sont classées (quelques centaines au plus), pas tout TikTok.

**Avec tes propres vidéos** : dépose-les dans le dossier `bestof\` (à la racine de `clipgod`), nommées dans
l'ordre du classement (`1 - ….mp4` = n°1). Un `@pseudo` dans le nom sert de crédit (`1 - @KaiCenat crash.mp4`).

```powershell
clipgod bestof --folder bestof --plan   # vérifie l'ordre et les durées
clipgod bestof --folder bestof          # -> output\bestof\TOP5_<date>.mp4 + .txt ; vidéos rangées dans bestof\publies\<date>\
```

Chaque vidéo est coupée à 30 s maximum (`bestof.folder_max_clip_seconds`). La tâche du matin utilise
automatiquement ce dossier dès qu'il contient au moins 2 vidéos.

**À la main**, sans tâche planifiée :

```powershell
clipgod discover --trending              # ce qui buzze : Shorts viraux du jour -> VODs d'origine
clipgod discover --trending --prepare 2  # … et prépare les 2 meilleures VODs pour Colab
clipgod bestof --plan                    # sélection du Top 5 sans montage
clipgod bestof                           # monte output\bestof\TOP5_<date>.mp4 + la légende .txt
```

**Comment le Top 5 est choisi** (`bestof` dans `config.yaml`) : les moments rendus depuis 48 h, une variante par
moment (le meilleur hook), sans cinématiques ni sources dans une autre langue, 2 moments maximum par streamer,
jamais un moment déjà utilisé dans un précédent Top 5. Compte à rebours #5 → #1, ~90 s, crédits dans la légende.
Il faut au moins 3 moments récents : avec 2 VODs par jour (3 moments chacune), c'est largement suffisant.

**Pourquoi les Shorts viraux ne sont pas republiés** : TikTok déclasse le « contenu non original » et les
créateurs signalent les re-uploads. ClipGod s'en sert uniquement comme signal pour trouver les VODs d'origine, puis
recoupe le moment lui-même (sous-titres, hook, classement, crédit) : c'est un montage transformé, qui t'appartient.

---

## 16. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `python` ou `clipgod` introuvable | PATH ou environnement non activé | Rouvrez le terminal, réactivez `.venv` (étape 4). |
| `config.yaml introuvable` | Commande lancée hors du dossier `clipgod` | `cd` dans le dossier, ou `clipgod --config chemin/config.yaml …`. |
| « Accès bloqué : ClipGod n'a pas terminé la procédure de validation » | Adresse absente des utilisateurs test | Étape 5.3 : ajoutez votre Gmail aux utilisateurs test. |
| Drive ou YouTube redemande l'autorisation chaque semaine, ou `invalid_grant` | Application OAuth en mode « Test » (jetons valables 7 jours) | Supprimez `secrets/drive_token.json` (ou `youtube_token.json`) et relancez ; pour de bon, passez l'application en production (étape 5.3). |
| `API Drive … HTTP 403` | API Drive non activée, ou mauvais compte | Étape 5.2 ; vérifiez que Colab et le laptop utilisent le même compte. |
| `Variable d'environnement … manquante` | `.env` incomplet | Étape 7 ; le nom de la variable est donné dans le message. |
| `discover` : « Aucune plateforme exploitable » | Clés YouTube/Twitch absentes ou invalides | Étapes 5.5 et 6 ; `clipgod discover -p twitch` pour tester une plateforme seule. |
| Kick ignoré avec un avertissement | Endpoints non officiels bloqués par Cloudflare | Normal de temps en temps ; YouTube et Twitch continuent. |
| `ffmpeg est requis` pendant `prepare` | ffmpeg absent du laptop | Étape 2, ou `clipgod prepare --download cloud …`. |
| Colab : « Aucun GPU » | Type d'exécution sans GPU, ou quota GPU épuisé | Étape 9.3 ; si le quota est épuisé, réessayez plus tard ou utilisez Kaggle. |
| Colab : `Unable to load libcudnn_ops.so.9` | Bibliothèques CUDA non trouvées | Exécution › Redémarrer la session, puis Tout exécuter. ClipGod précharge ces bibliothèques automatiquement. |
| Le job échoue avec « Sign in to confirm you're not a bot » | YouTube bloque le téléchargement depuis Colab | `clipgod prepare --download local --force --source <url>` : le laptop télécharge la VOD. |
| `clipgod sync` : job `failed` | Erreur côté notebook (affichée) | Lisez l'erreur ; un job est retenté automatiquement une fois avant d'être marqué `failed`. |
| Vidéos YouTube toujours privées en `--live` | Projet Google Cloud non audité | Demandez l'audit de l'API YouTube (formulaire gratuit de Google), ou publiez via Upload-Post. |
| `Quota YouTube insuffisant` | 10 000 unités/jour atteintes | Remise à zéro à minuit, heure du Pacifique (9 h à Paris). `publish` continue sur les autres plateformes. |
| `monetize` ne trouve aucune campagne | Le site Whop a changé | Ajoutez vos campagnes dans `data/whop_campaigns.yaml` (étape 12). |
| `learn` : « pas de réentraînement » | Moins de 10 clips mesurés à 24 h | Normal au début : publiez, attendez 24 h, relancez `learn`. |

Les journaux détaillés sont dans **`logs/clipgod.log`** (une ligne JSON par événement). Ajoutez `-v` à
n'importe quelle commande pour plus de détails : `clipgod -v sync`.

### Surveillance des lives (`watch`, optionnel)

```bash
pip install -e ".[watch]"                                 # streamlink + websockets
clipgod watch https://www.twitch.tv/<chaine> --force
```

| Symptôme | Solution |
|---|---|
| « watch est désactivé » | Ajoutez `--force`, ou mettez `watch.disable-watch: false` dans `config.yaml`. |
| « streamlink n'est pas installé » | `pip install -e ".[watch]"` (environnement activé). |
| Le chat Kick ne se connecte pas | Kick a changé sa clé de WebSocket : mettez la nouvelle dans `watch.kick.pusher_key`. Le buffer et la capture manuelle (`c`) fonctionnent quand même. |
| Chat YouTube « quota YouTube bas » | La lecture du chat coûte 5 unités ; elle se suspend pour garder de quoi publier. |
| Connexion lente / coupures | `--quality 480p` divise le débit par deux environ. |
