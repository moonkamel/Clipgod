"""Module 2 — watch (laptop/VPS) : surveille un live et capture automatiquement les moments de hype.

    clipgod watch <url du live> [--force]

Fonctionnement :
    - **buffer roulant** : ``streamlink --stdout <url> | ffmpeg -f segment`` écrit le live en segments de
      ``segment_seconds`` (60 s) dans ``cache/buffer/`` ; seuls les ``buffer_minutes`` (8) derniers sont gardés
      (plus ceux verrouillés par une capture en cours) -> espace disque constant.
    - **chat** analysé toutes les ``chat_poll_seconds`` (2 s) : IRC Twitch (anonyme), WebSocket Kick (Pusher public),
      API YouTube Live Chat (quota surveillé, 5 unités par lecture).
    - **score de hype** = somme pondérée des écarts à la normale (z-scores sur ``baseline_window_seconds``) de :
      vitesse du chat, nouveaux chatteurs, proportion de messages réactionnels (MAJUSCULES, emojis, WTF, NO WAY,
      CLIP THAT…). Déclenchement si score > moyenne + ``sigma_threshold`` x écart-type du score.
    - **capture** : les 8 minutes du buffer + les ``post_trigger_minutes`` (2) suivantes, assemblées sans
      réencodage dans ``inbox/raw_<AAAAMMJJ_HHMMSS>.mp4``, prêtes pour ``clipgod prepare``.
    - **robustesse** : si le stream coupe, ``max_attempts`` (3) reconnexions avec attente exponentielle
      (5 s, 10 s, 20 s) ; ``cooldown_seconds`` (120) entre deux déclenchements.

Désactivé par défaut (``watch.disable-watch: true``) : il télécharge le live en continu.
Touches : ``c`` capture manuelle · ``p`` envoie la dernière capture au notebook (prepare) · ``q`` / Ctrl+C quitter.
"""

from __future__ import annotations

import contextlib
import json
import queue
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import get, get_env, resolve_path
from core.logger import get_logger
from core.storage import utc_now

log = get_logger(__name__)

EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")
SEG_RE = re.compile(r"seg_(\d{5,})\.ts$")


class WatchError(RuntimeError):
    """Live introuvable, outils manquants, stream terminé…"""


# ===========================================================================
# Plateformes
# ===========================================================================
@dataclass(frozen=True)
class LiveTarget:
    platform: str          # twitch | youtube | kick
    channel: str           # login / slug / id vidéo
    url: str


def parse_live_url(url: str) -> LiveTarget:
    u = url.strip()
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)/?$", u)
    if m and m.group(1).lower() not in ("videos", "directory"):
        return LiveTarget("twitch", m.group(1).lower(), u)
    m = re.search(r"kick\.com/([A-Za-z0-9_-]+)/?$", u)
    if m:
        return LiveTarget("kick", m.group(1).lower(), u)
    m = re.search(r"(?:youtube\.com/(?:watch\?v=|live/)|youtu\.be/)([A-Za-z0-9_-]{11})", u)
    if m:
        return LiveTarget("youtube", m.group(1), u)
    m = re.search(r"youtube\.com/(@[A-Za-z0-9_.-]+)(?:/live)?/?$", u)
    if m:
        return LiveTarget("youtube", m.group(1), u.rstrip("/") + ("" if u.rstrip("/").endswith("/live") else "/live"))
    raise WatchError(f"URL de live non reconnue : {url} (Twitch, Kick ou YouTube).")


# ===========================================================================
# Chat
# ===========================================================================
@dataclass
class ChatMessage:
    user: str
    text: str
    ts: float = field(default_factory=time.time)


class ChatSource:
    """Lit un chat dans un thread et pousse des :class:`ChatMessage` dans ``self.queue``."""

    name = "chat"

    def __init__(self) -> None:
        self.queue: queue.Queue[ChatMessage] = queue.Queue()
        self.events: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._guarded, name=self.name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _guarded(self) -> None:
        delay = 2.0
        while not self._stop.is_set():
            try:
                self.run()
                delay = 2.0
            except Exception as exc:  # noqa: BLE001 — on reconnecte le chat sans arrêter le buffer
                self.events.put(f"chat {self.name} déconnecté ({exc}) · reconnexion dans {delay:.0f} s")
                log.warning("chat %s : %s", self.name, exc)
            if self._stop.wait(delay):
                return
            delay = min(60.0, delay * 2)

    def drain(self) -> list[ChatMessage]:
        out = []
        with contextlib.suppress(queue.Empty):
            while True:
                out.append(self.queue.get_nowait())
        return out

    def run(self) -> None:  # pragma: no cover — implémenté par les sous-classes
        raise NotImplementedError


IRC_PRIVMSG = re.compile(r"^(?:@\S+ )?:(\w+)!\S+ PRIVMSG #\w+ :(.*)$")


def parse_irc_line(line: str) -> ChatMessage | str | None:
    """Ligne IRC Twitch -> ChatMessage, ``"PING"`` (à renvoyer en PONG) ou None."""
    line = line.rstrip("\r\n")
    if line.startswith("PING"):
        return "PING"
    m = IRC_PRIVMSG.match(line)
    return ChatMessage(m.group(1), m.group(2)) if m else None


class TwitchChat(ChatSource):
    """IRC Twitch en lecture anonyme (justinfan) : aucune clé nécessaire."""

    name = "twitch-irc"

    def __init__(self, channel: str, *, host: str = "irc.chat.twitch.tv", port: int = 6667,
                 connect: Callable[..., Any] = socket.create_connection):
        super().__init__()
        self.channel, self.host, self.port, self._connect = channel.lower(), host, port, connect

    def run(self) -> None:
        sock = self._connect((self.host, self.port), timeout=10)
        try:
            nick = f"justinfan{random.randint(10000, 99999)}"
            sock.sendall(f"PASS SCHMOOPIIE\r\nNICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
            self.events.put(f"chat IRC connecté · #{self.channel}")
            sock.settimeout(1.0)
            buf = ""
            last_data = time.time()
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except TimeoutError:
                    if time.time() - last_data > 360:
                        raise ConnectionError("aucune donnée depuis 6 min") from None
                    continue
                if not chunk:
                    raise ConnectionError("connexion fermée par Twitch")
                last_data = time.time()
                buf += chunk.decode("utf-8", errors="replace")
                *lines, buf = buf.split("\n")
                for line in lines:
                    parsed = parse_irc_line(line)
                    if parsed == "PING":
                        sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                    elif isinstance(parsed, ChatMessage):
                        self.queue.put(parsed)
        finally:
            with contextlib.suppress(OSError):
                sock.close()


def parse_kick_event(raw: str) -> ChatMessage | None:
    """Message Pusher de Kick -> ChatMessage (événement ``App\\Events\\ChatMessageEvent``)."""
    try:
        frame = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "ChatMessage" not in str(frame.get("event", "")):
        return None
    data = frame.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    user = (data.get("sender") or {}).get("username") or "?"
    return ChatMessage(str(user), str(data.get("content", "")))


class KickChat(ChatSource):
    """WebSocket public de Kick (Pusher). L'id du salon vient de l'endpoint JSON public de la chaîne."""

    name = "kick-ws"

    def __init__(self, slug: str, *, pusher_key: str, pusher_cluster: str = "us2", http_client: Any = None):
        super().__init__()
        self.slug, self.key, self.cluster, self._http = slug, pusher_key, pusher_cluster, http_client

    def chatroom_id(self) -> int:
        import httpx

        from modules.discover import BROWSER_HEADERS, KICK_API

        client = self._http or httpx.Client(timeout=15, headers=BROWSER_HEADERS, follow_redirects=True)
        try:
            resp = client.get(f"{KICK_API}/channels/{self.slug}")
            resp.raise_for_status()
            return int(resp.json()["chatroom"]["id"])
        finally:
            if self._http is None:
                client.close()

    def run(self) -> None:
        from websockets.sync.client import connect

        room = self.chatroom_id()
        url = f"wss://ws-{self.cluster}.pusher.com/app/{self.key}?protocol=7&client=js&version=8.4.0&flash=false"
        with connect(url, open_timeout=15) as ws:
            ws.send(json.dumps({"event": "pusher:subscribe", "data": {"auth": "", "channel": f"chatrooms.{room}.v2"}}))
            self.events.put(f"chat Kick connecté · salon {room}")
            while not self._stop.is_set():
                try:
                    raw = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                if '"pusher:ping"' in raw:
                    ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                    continue
                msg = parse_kick_event(raw)
                if msg:
                    self.queue.put(msg)


class YouTubeChat(ChatSource):
    """API YouTube Live Chat (clé ``YOUTUBE_API_KEY``). 5 unités de quota par lecture : intervalle >= 10 s."""

    name = "youtube-chat"

    def __init__(self, cfg: Mapping[str, Any], target: LiveTarget, *, service: Any = None):
        super().__init__()
        self.cfg, self.target, self._service = cfg, target, service
        self.min_interval = float(get(cfg, "watch.youtube_chat_min_poll_seconds", 10))

    def _video_id(self) -> str:
        if not self.target.channel.startswith("@"):
            return self.target.channel
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(self.target.url, download=False, process=False)
        return str(info["id"])

    def run(self) -> None:
        from core.quota import YouTubeQuota

        quota = YouTubeQuota.from_config(self.cfg)
        if self._service is None:
            key = get_env(get(self.cfg, "discover.env.youtube_api_key", "YOUTUBE_API_KEY"), required=True)
            from googleapiclient.discovery import build

            self._service = build("youtube", "v3", developerKey=key, cache_discovery=False)
        quota.spend(1, "videos.list")
        items = self._service.videos().list(part="liveStreamingDetails", id=self._video_id()).execute().get("items", [])
        chat_id = ((items[0] if items else {}).get("liveStreamingDetails") or {}).get("activeLiveChatId")
        if not chat_id:
            raise WatchError("pas de chat en direct actif sur cette vidéo")
        self.events.put("chat YouTube connecté")
        token = None
        while not self._stop.is_set():
            if not quota.can_spend(5 + 1600):          # on laisse toujours de quoi faire un upload
                self.events.put("quota YouTube bas : lecture du chat suspendue")
                self._stop.wait(300)
                continue
            quota.spend(5, "liveChatMessages.list")
            resp = self._service.liveChatMessages().list(liveChatId=chat_id, part="snippet,authorDetails",
                                                         pageToken=token, maxResults=2000).execute()
            token = resp.get("nextPageToken")
            for it in resp.get("items", []):
                self.queue.put(ChatMessage((it.get("authorDetails") or {}).get("displayName", "?"),
                                           (it.get("snippet") or {}).get("displayMessage", "")))
            wait = max(self.min_interval, float(resp.get("pollingIntervalMillis", 0)) / 1000)
            self._stop.wait(wait)


# ===========================================================================
# Détection de hype (logique pure)
# ===========================================================================
def is_reaction(text: str, patterns: Sequence[str]) -> bool:
    """Message réactionnel : MAJUSCULES, emoji, ou expression de la liste (WTF, NO WAY, CLIP THAT, KEKW…)."""
    t = text.strip()
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        return True
    if EMOJI_RE.search(t):
        return True
    low = t.lower()
    return any(re.search(rf"(?<!\w){re.escape(p.lower())}(?!\w)", low) or (not p.isascii() and p in t)
               for p in patterns)


@dataclass
class HypeTick:
    rate: int
    new_chatters: int
    reaction_ratio: float
    ratio_to_mean: float
    score: float
    mean: float
    threshold: float
    sigma: float                 # écart du score à sa moyenne, en écarts-types
    warmed_up: bool
    above: bool                  # score > seuil (le Watcher décide du déclenchement : cooldown…)


class HypeDetector:
    """Score de hype par tick : z-scores pondérés vitesse / nouveaux chatteurs / réactions vs la normale récente."""

    def __init__(self, hype_cfg: Mapping[str, Any], *, tick_seconds: float = 2.0):
        window = int(float(hype_cfg.get("baseline_window_seconds", 300)) / tick_seconds)
        self.warmup = int(float(hype_cfg.get("min_warmup_seconds", 60)) / tick_seconds)
        self.sigma = float(hype_cfg.get("sigma_threshold", 2.0))
        w = hype_cfg.get("weights") or {}
        self.w = (float(w.get("chat_velocity", 0.5)), float(w.get("new_chatters", 0.2)),
                  float(w.get("reaction_ratio", 0.3)))
        self.patterns = list(hype_cfg.get("reaction_patterns") or [])
        self.hist: tuple[deque[float], deque[float], deque[float]] = (deque(maxlen=window), deque(maxlen=window),
                                                                     deque(maxlen=window))
        self.scores: deque[float] = deque(maxlen=window)
        self.seen: set[str] = set()

    @staticmethod
    def _stats(values: Iterable[float]) -> tuple[float, float]:
        vals = list(values)
        if not vals:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def update(self, messages: Sequence[ChatMessage]) -> HypeTick:
        rate = len(messages)
        users = {m.user.lower() for m in messages}
        new = len(users - self.seen)
        self.seen |= users
        reactions = sum(is_reaction(m.text, self.patterns) for m in messages)
        ratio = reactions / rate if rate else 0.0
        feats = (float(rate), float(new), ratio)
        floors = (lambda m: 0.5 + 0.1 * m, lambda m: 0.5 + 0.1 * m, lambda m: 0.05)
        z = []
        for x, hist, floor in zip(feats, self.hist, floors, strict=True):
            m, sd = self._stats(hist)
            z.append((x - m) / max(sd, floor(m)) if hist else 0.0)
        score = sum(wi * zi for wi, zi in zip(self.w, z, strict=True))
        sm, ssd = self._stats(self.scores)
        ssd = max(ssd, 0.25)
        threshold = sm + self.sigma * ssd
        warmed = len(self.scores) >= self.warmup
        mean_rate = self._stats(self.hist[0])[0]
        tick = HypeTick(rate, new, ratio, rate / mean_rate if mean_rate else 1.0, score, sm, threshold,
                        (score - sm) / ssd, warmed, warmed and score > threshold)
        for hist, x in zip(self.hist, feats, strict=True):
            hist.append(x)
        self.scores.append(score)
        return tick


# ===========================================================================
# Buffer (streamlink | ffmpeg segment)
# ===========================================================================
def list_segments(buffer_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in buffer_dir.glob("seg_*.ts"):
        m = SEG_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


class BufferRecorder:
    """Enregistre le live en segments ``seg_NNNNN.ts`` et ne garde que les ``keep`` derniers (hors verrous)."""

    def __init__(self, url: str, buffer_dir: Path, *, quality: str = "720p,best", segment_seconds: int = 60,
                 keep: int = 8, popen: Callable[..., Any] = subprocess.Popen):
        self.url, self.dir, self.quality = url, Path(buffer_dir), quality
        self.segment_seconds, self.keep, self._popen = int(segment_seconds), int(keep), popen
        self.procs: list[Any] = []
        self.started_at = 0.0
        self.dir.mkdir(parents=True, exist_ok=True)

    def next_index(self) -> int:
        segs = list_segments(self.dir)
        return segs[-1][0] + 1 if segs else 0

    def start(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise WatchError("ffmpeg introuvable (voir docs/INSTALLATION.md, étape 2).")
        sl = self._popen([sys.executable, "-m", "streamlink", "--stdout", "--retry-open", "2", self.url, self.quality],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ff = self._popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-map", "0", "-c", "copy",
                          "-f", "segment", "-segment_time", str(self.segment_seconds), "-reset_timestamps", "1",
                          "-segment_start_number", str(self.next_index()), str(self.dir / "seg_%05d.ts")],
                         stdin=sl.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if getattr(sl, "stdout", None) is not None:
            with contextlib.suppress(Exception):
                sl.stdout.close()           # ffmpeg détient désormais le tube
        self.procs = [sl, ff]
        self.started_at = time.monotonic()

    def alive(self) -> bool:
        return bool(self.procs) and all(p.poll() is None for p in self.procs)

    def stop(self) -> None:
        for p in reversed(self.procs):
            with contextlib.suppress(Exception):
                p.terminate()
                p.wait(timeout=5)
        self.procs = []

    def current_index(self) -> int | None:
        segs = list_segments(self.dir)
        return segs[-1][0] if segs else None

    def prune(self, locked: set[int]) -> list[int]:
        """Supprime les segments plus vieux que le buffer, sauf ceux verrouillés par une capture."""
        segs = list_segments(self.dir)
        if not segs:
            return []
        cutoff = segs[-1][0] - self.keep
        removed = []
        for idx, p in segs:
            if idx <= cutoff and idx not in locked:
                with contextlib.suppress(OSError):
                    p.unlink()
                    removed.append(idx)
        return removed

    def disk_usage(self) -> int:
        return sum(p.stat().st_size for _, p in list_segments(self.dir) if p.exists())


def assemble(segments: Sequence[Path], out: Path) -> Path:
    """Concatène des segments .ts sans réencodage -> MP4 (+faststart)."""
    if not segments:
        raise WatchError("aucun segment à assembler")
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8")
    tmp = out.with_suffix(".part.mp4")
    proc = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                           "-safe", "0", "-i", str(listing), "-c", "copy", "-bsf:a", "aac_adtstoasc",
                           "-movflags", "+faststart", str(tmp)], capture_output=True, text=True, check=False)
    listing.unlink(missing_ok=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise WatchError(f"assemblage impossible : {proc.stderr.strip()[-300:]}")
    tmp.replace(out)
    return out


# ===========================================================================
# Orchestration
# ===========================================================================
@dataclass
class Capture:
    file: Path
    first: int
    last: int
    reason: str
    started: float
    status: str = "recording"         # recording | ready | failed | sent
    error: str | None = None

    @property
    def indices(self) -> set[int]:
        return set(range(self.first, self.last + 1))


@dataclass
class WatchState:
    target: LiveTarget
    status: str = "starting"          # live | rec | reconnecting | ended
    started: float = field(default_factory=time.monotonic)
    scores: deque = field(default_factory=lambda: deque(maxlen=90))
    thresholds: deque = field(default_factory=lambda: deque(maxlen=90))
    triggers: deque = field(default_factory=lambda: deque(maxlen=90))   # bool par tick
    last_tick: HypeTick | None = None
    chat: deque = field(default_factory=lambda: deque(maxlen=40))
    events: deque = field(default_factory=lambda: deque(maxlen=12))
    captures: list[Capture] = field(default_factory=list)
    reconnect_attempt: int = 0


class Watcher:
    """Boucle principale. ``tick()`` est appelée toutes les ``chat_poll_seconds`` (testable sans threads)."""

    def __init__(self, cfg: Mapping[str, Any], target: LiveTarget, *, recorder: BufferRecorder,
                 chat: ChatSource | None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, assembler: Callable[[Sequence[Path], Path], Path] = assemble,
                 now: Callable[[], datetime] = utc_now):
        w = cfg.get("watch") or {}
        self.cfg, self.target, self.recorder, self.chat = cfg, target, recorder, chat
        self.clock, self.sleep, self.assembler, self.now = clock, sleep, assembler, now
        self.tick_seconds = float(w.get("chat_poll_seconds", 2))
        self.detector = HypeDetector(w.get("hype") or {}, tick_seconds=self.tick_seconds)
        self.buffer_segments = max(1, round(float(w.get("buffer_minutes", 8)) * 60 / recorder.segment_seconds))
        self.post_segments = max(0, round(float(w.get("post_trigger_minutes", 2)) * 60 / recorder.segment_seconds))
        self.cooldown = float(w.get("cooldown_seconds", 120))
        rc = w.get("reconnect") or {}
        self.max_attempts, self.backoff = int(rc.get("max_attempts", 3)), float(rc.get("backoff_base_seconds", 5))
        self.inbox = resolve_path(cfg, get(cfg, "paths.inbox", "inbox"))
        self.state = WatchState(target)
        self._last_trigger = -1e9
        self._retry_at: float | None = None
        self._stop = threading.Event()
        self.on_capture_ready: Callable[[Capture], None] | None = None

    # -- événements -------------------------------------------------------
    def event(self, text: str, style: str = "") -> None:
        stamp = self.now().astimezone().strftime("%H:%M:%S")
        self.state.events.appendleft((stamp, text, style))
        log.info(text)

    # -- cycle de vie -----------------------------------------------------
    def start(self) -> None:
        self.recorder.start()
        self.state.status = "live"
        self.event(f"streamlink connecté · {self.recorder.quality} · segments de {self.recorder.segment_seconds} s", "green")
        if self.chat:
            self.chat.start()

    def stop(self) -> None:
        self._stop.set()
        if self.chat:
            self.chat.stop()
        self.recorder.stop()
        for cap in self.state.captures:          # finalise ce qui peut l'être
            if cap.status == "recording":
                self._finish(cap, partial=True)

    def run(self, render: Callable[[WatchState], None] | None = None, keys: queue.Queue[str] | None = None) -> None:
        self.start()
        next_at = self.clock()
        try:
            while not self._stop.is_set() and self.state.status != "ended":
                self.tick()
                if keys is not None:
                    with contextlib.suppress(queue.Empty):
                        while True:
                            self.handle_key(keys.get_nowait())
                if render:
                    render(self.state)
                next_at += self.tick_seconds
                self.sleep(max(0.0, next_at - self.clock()))
        finally:
            self.stop()

    # -- tick -------------------------------------------------------------
    def tick(self) -> HypeTick | None:
        now = self.clock()
        if self.chat:
            while not self.chat.events.empty():
                self.event(self.chat.events.get_nowait(), "cyan")
        if not self._supervise(now):
            return None
        messages = self.chat.drain() if self.chat else []
        for m in messages[-12:]:
            self.state.chat.append((m.user, m.text, is_reaction(m.text, self.detector.patterns)))
        hype = self.detector.update(messages)
        st = self.state
        st.last_tick = hype
        st.scores.append(hype.score)
        st.thresholds.append(hype.threshold)
        fired = False
        if hype.above and not self._busy(now):
            self.trigger(f"pic de hype {hype.sigma:.1f}σ (score {hype.score:.2f} > seuil {hype.threshold:.2f})",
                         label=f"{hype.sigma:.1f}σ")
            fired = True
        st.triggers.append(fired)
        self._progress_captures()
        self.recorder.prune(self._locked())
        return hype

    def _busy(self, now: float) -> bool:
        return now - self._last_trigger < self.cooldown or any(c.status == "recording" for c in self.state.captures)

    def _locked(self) -> set[int]:
        out: set[int] = set()
        for c in self.state.captures:
            if c.status == "recording":
                out |= c.indices
        return out

    def _supervise(self, now: float) -> bool:
        """Surveille le pipeline streamlink/ffmpeg ; reconnexion avec attente exponentielle."""
        st = self.state
        if self.recorder.alive():
            if st.status == "reconnecting":
                self.event(f"reconnecté au live (essai {st.reconnect_attempt}/{self.max_attempts}) · buffer repris", "green")
                st.status = "live"
            elif st.reconnect_attempt and now - self.recorder.started_at > 60:
                st.reconnect_attempt = 0
            return True
        if self._retry_at is None:
            if st.reconnect_attempt >= self.max_attempts:
                st.status = "ended"
                self.event("stream terminé (plus de reconnexion possible) · arrêt", "red")
                return False
            st.reconnect_attempt += 1
            wait = self.backoff * 2 ** (st.reconnect_attempt - 1)
            self._retry_at = now + wait
            st.status = "reconnecting"
            self.event(f"stream interrompu · reconnexion {st.reconnect_attempt}/{self.max_attempts} dans {wait:.0f} s", "red")
            return False
        if now >= self._retry_at:
            self._retry_at = None
            self.recorder.stop()
            try:
                self.recorder.start()
            except Exception as exc:  # noqa: BLE001
                self.event(f"échec de la reconnexion : {exc}", "red")
        return False

    # -- captures -----------------------------------------------------------
    def trigger(self, reason: str, *, label: str = "manuel") -> Capture | None:
        current = self.recorder.current_index()
        if current is None:
            self.event("capture impossible : buffer encore vide", "yellow")
            return None
        stamp = self.now().astimezone().strftime("%Y%m%d_%H%M%S")
        cap = Capture(self.inbox / f"raw_{stamp}.mp4", max(0, current - self.buffer_segments + 1),
                      current + self.post_segments, label, self.clock())
        self.state.captures.append(cap)
        self._last_trigger = self.clock()
        self.state.status = "rec"
        self.event(reason, "yellow")
        self.event(f"{self.buffer_segments} min du buffer verrouillées · enregistrement des "
                   f"{self.post_segments} min suivantes", "cyan")
        return cap

    def _progress_captures(self) -> None:
        current = self.recorder.current_index()
        for cap in self.state.captures:
            if cap.status == "recording" and current is not None and current > cap.last:   # dernier segment fermé
                self._finish(cap)
        if self.state.status == "rec" and not any(c.status == "recording" for c in self.state.captures):
            self.state.status = "live"

    def _finish(self, cap: Capture, *, partial: bool = False) -> None:
        """Assemble la capture ; ``partial`` = arrêt avant la fin des 2 min (ffmpeg déjà arrêté, segments fermés)."""
        segs = [p for i, p in list_segments(self.recorder.dir) if cap.first <= i <= cap.last]
        try:
            self.assembler(segs, cap.file)
            cap.status = "ready"
            self.event(f"capture écrite : {cap.file.name} ({len(segs)} min) · prête pour clipgod prepare", "green")
            if self.on_capture_ready:
                self.on_capture_ready(cap)
        except Exception as exc:  # noqa: BLE001
            cap.status, cap.error = "failed", str(exc)
            self.event(f"capture {cap.file.name} en échec : {exc}", "red")

    def handle_key(self, key: str) -> None:
        k = key.lower()
        if k == "c":
            if self._busy(self.clock()):
                self.event("capture déjà en cours ou cooldown actif", "yellow")
            else:
                self.trigger("capture manuelle demandée (touche c)")
        elif k == "p":
            ready = [c for c in self.state.captures if c.status == "ready"]
            if not ready:
                self.event("aucune capture prête à envoyer", "yellow")
            else:
                threading.Thread(target=self._send, args=(ready[-1],), daemon=True).start()
        elif k == "q":
            self._stop.set()

    def _send(self, cap: Capture) -> None:
        """Touche p : dépose la capture comme un job (même chose que clipgod prepare --source)."""
        try:
            from core.drive import get_drive
            from modules.prepare import prepare

            job = prepare(str(cap.file), self.cfg, get_drive(self.cfg), streamer=self.target.channel.lstrip("@"))
            cap.status = "sent"
            self.event(f"{cap.file.name} envoyée · job {job['job_id']} · ouvrez le notebook Colab", "green")
        except Exception as exc:  # noqa: BLE001
            self.event(f"envoi de {cap.file.name} impossible : {exc}", "red")


# ===========================================================================
# Affichage (rich) et clavier
# ===========================================================================
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], top: float, bottom: float) -> str:
    span = max(1e-9, top - bottom)
    return "".join(SPARK[min(7, max(0, int((v - bottom) / span * 7.999)))] for v in values)


def render_state(state: WatchState, recorder: BufferRecorder) -> Any:
    """Écran de la maquette, rendu avec rich."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    pills = {"live": ("● LIVE", "bold white on red"), "rec": ("● REC +2 min", "bold black on yellow"),
             "reconnecting": ("○ COUPÉ", "bold white on grey30"), "ended": ("■ TERMINÉ", "bold white on grey30"),
             "starting": ("… DÉMARRAGE", "bold white on grey30")}
    label, style = pills.get(state.status, pills["starting"])
    up = int(time.monotonic() - state.started)
    head = Text.assemble((f" {label} ", style), "  chaîne ", (state.target.channel, "bold"), "  plateforme ",
                         (state.target.platform, "bold"), "  surveillé depuis ",
                         (f"{up // 3600:02d}:{up // 60 % 60:02d}:{up % 60:02d}", "bold"),
                         "  buffer ", (f"{recorder.disk_usage() / 1e6:.0f} Mo", "bold"))

    segs = list_segments(recorder.dir)[-recorder.keep:]
    locked = set()
    for c in state.captures:
        if c.status == "recording":
            locked |= c.indices
    buf = Text()
    for i, _ in segs:
        buf.append(f" seg {i:05d} ", "black on yellow" if i in locked else "black on cyan")
        buf.append(" ")
    buffer_panel = Panel(buf or Text("en attente du premier segment…", style="dim"),
                         title="[cyan]Buffer roulant[/]", title_align="left")

    t = state.last_tick
    scores = list(state.scores)
    if scores:
        top = max(max(scores), max(state.thresholds or [0])) or 1
        bottom = min(min(scores), 0)
        spark = Text(sparkline(scores, top, bottom), style="magenta")
        marks = Text("".join("▲" if f else " " for f in state.triggers), style="yellow")
        thr_line = f"seuil = moyenne + 2σ = {t.threshold:.2f}" if t else ""
    else:
        spark, marks, thr_line = Text("calibrage du chat…", style="dim"), Text(""), ""
    metrics = Table.grid(padding=(0, 3))
    for _ in range(4):
        metrics.add_column()
    if t:
        metrics.add_row(f"messages/2 s [bold]{t.rate}[/]", f"vs moyenne [bold]×{t.ratio_to_mean:.1f}[/]",
                        f"nouveaux chatteurs [bold]{t.new_chatters}[/]", f"réactions [bold]{t.reaction_ratio:.0%}[/]")
    hype_panel = Panel(Group(spark, marks, Text(thr_line + ("  · calibrage" if t and not t.warmed_up else ""),
                                                style="dim"), metrics),
                       title="[cyan]Score de hype[/]", title_align="left")

    chat = Text()
    for user, text, react in list(state.chat)[-14:]:
        chat.append(user, "bold cyan")
        chat.append(": ")
        chat.append(text[:70] + "\n", "bold yellow" if react else "")
    chat_panel = Panel(chat or Text("…", style="dim"), title="[cyan]Chat[/]", title_align="left")

    ev = Text()
    for stamp, text, style in list(state.events)[:7]:
        ev.append(stamp + " ", "dim")
        ev.append(text + "\n", style)
    caps = Text()
    for c in state.captures[-4:][::-1]:
        caps.append(f"inbox/{c.file.name}  ", "bold")
        caps.append({"recording": "enregistre", "ready": "prête", "failed": "échec", "sent": "envoyée"}[c.status] + "\n",
                    {"recording": "yellow", "ready": "green", "failed": "red", "sent": "cyan"}[c.status])
    grid = Table.grid(expand=True)
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(Group(buffer_panel, hype_panel), chat_panel)
    grid.add_row(Panel(ev, title="[cyan]Événements[/]", title_align="left"),
                 Panel(caps or Text("aucune capture", style="dim"), title="[cyan]Captures → inbox/[/]", title_align="left"))
    keys = Text("c capturer maintenant · p envoyer la dernière capture (prepare) · q / Ctrl+C arrêter", style="dim")
    return Group(head, grid, keys)


class KeyReader:
    """Lit les touches sans bloquer (Windows : msvcrt ; macOS/Linux : terminal en mode cbreak)."""

    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._restore: Callable[[], None] | None = None

    def start(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._restore:
            self._restore()

    def _loop(self) -> None:
        if sys.platform.startswith("win"):
            import msvcrt

            while not self._stop.is_set():
                if msvcrt.kbhit():
                    self.queue.put(msvcrt.getwch())
                time.sleep(0.05)
            return
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)
        tty.setcbreak(fd)
        try:
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    self.queue.put(sys.stdin.read(1))
        finally:
            self._restore()


def build_chat(cfg: Mapping[str, Any], target: LiveTarget) -> ChatSource:
    w = cfg.get("watch") or {}
    if target.platform == "twitch":
        return TwitchChat(target.channel)
    if target.platform == "kick":
        k = w.get("kick") or {}
        return KickChat(target.channel, pusher_key=str(k.get("pusher_key", "32cbd69e4b950bf97679")),
                        pusher_cluster=str(k.get("pusher_cluster", "us2")))
    return YouTubeChat(cfg, target)


def watch(cfg: Mapping[str, Any], url: str, *, quality: str | None = None, with_chat: bool = True,
          auto_prepare: bool | None = None, console: Any = None) -> WatchState:
    """Point d'entrée de ``clipgod watch`` : écran rich en direct jusqu'à Ctrl+C / q / fin du stream."""
    from rich.live import Live

    w = cfg.get("watch") or {}
    target = parse_live_url(url)
    if shutil.which("ffmpeg") is None:
        raise WatchError("ffmpeg introuvable (voir docs/INSTALLATION.md, étape 2).")
    try:
        import streamlink  # noqa: F401
    except ImportError as exc:
        raise WatchError("streamlink n'est pas installé : pip install -e \".[watch]\"") from exc
    buffer_dir = resolve_path(cfg, get(cfg, "paths.cache", "cache")) / "buffer" / f"{target.platform}_{target.channel.lstrip('@')}"
    recorder = BufferRecorder(target.url, buffer_dir, quality=quality or str(w.get("quality", "720p,best")),
                              segment_seconds=int(w.get("segment_seconds", 60)),
                              keep=round(float(w.get("buffer_minutes", 8)) * 60 / int(w.get("segment_seconds", 60))))
    watcher = Watcher(cfg, target, recorder=recorder, chat=build_chat(cfg, target) if with_chat else None)
    if auto_prepare if auto_prepare is not None else bool(w.get("auto_prepare", False)):
        watcher.on_capture_ready = lambda cap: threading.Thread(target=watcher._send, args=(cap,), daemon=True).start()
    keys = KeyReader()
    keys.start()
    try:
        with Live(render_state(watcher.state, recorder), console=console, refresh_per_second=4,
                  transient=False) as live:
            watcher.run(render=lambda st: live.update(render_state(st, recorder)), keys=keys.queue)
    except KeyboardInterrupt:
        watcher.stop()
    finally:
        keys.stop()
        if not w.get("keep_buffer_on_exit", False):
            shutil.rmtree(buffer_dir, ignore_errors=True)
    return watcher.state
