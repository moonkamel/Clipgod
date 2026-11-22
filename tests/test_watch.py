import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.config import load_config
from modules import watch as W

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")
PATTERNS = ["WTF", "NO WAY", "CLIP THAT", "KEKW", "💀"]


def test_parse_live_url():
    assert W.parse_live_url("https://www.twitch.tv/NightOwl_RP") == W.LiveTarget("twitch", "nightowl_rp", "https://www.twitch.tv/NightOwl_RP")
    assert W.parse_live_url("https://kick.com/xqc").platform == "kick"
    assert W.parse_live_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ").channel == "dQw4w9WgXcQ"
    yt = W.parse_live_url("https://www.youtube.com/@somechannel")
    assert yt.channel == "@somechannel" and yt.url.endswith("/live")
    with pytest.raises(W.WatchError):
        W.parse_live_url("https://www.twitch.tv/videos/123")


def test_is_reaction():
    assert W.is_reaction("NO WAY bro", PATTERNS) and W.is_reaction("wtf just happened", PATTERNS)
    assert W.is_reaction("HE DID IT", PATTERNS) and W.is_reaction("lol 💀", PATTERNS)
    assert W.is_reaction("KEKW", PATTERNS)
    assert not W.is_reaction("where is this mission", PATTERNS) and not W.is_reaction("ok", PATTERNS)
    assert not W.is_reaction("nowayyy", ["NO WAY"])


def test_irc_and_kick_parsing():
    msg = W.parse_irc_line(":mossy_pixel!mossy_pixel@mossy_pixel.tmi.twitch.tv PRIVMSG #chan :NO WAY 💀\r\n")
    assert msg.user == "mossy_pixel" and msg.text == "NO WAY 💀"
    assert W.parse_irc_line("PING :tmi.twitch.tv") == "PING" and W.parse_irc_line(":tmi.twitch.tv 001 x :Welcome") is None
    tagged = W.parse_irc_line("@badge-info=;color=#FF0000 :u1!u1@u1.tmi.twitch.tv PRIVMSG #chan :hello")
    assert tagged.user == "u1" and tagged.text == "hello"
    frame = json.dumps({"event": "App\\Events\\ChatMessageEvent",
                        "data": json.dumps({"content": "CLIP THAT", "sender": {"username": "kickfan"}})})
    km = W.parse_kick_event(frame)
    assert km.user == "kickfan" and km.text == "CLIP THAT"
    assert W.parse_kick_event('{"event":"pusher:connection_established"}') is None and W.parse_kick_event("nope") is None


class FakeSock:
    def __init__(self, chunks):
        self.chunks, self.sent = list(chunks), []

    def sendall(self, data):
        self.sent.append(data.decode())

    def settimeout(self, t):
        pass

    def recv(self, n):
        if self.chunks:
            return self.chunks.pop(0)
        return b""          # fermeture -> ConnectionError

    def close(self):
        pass


def test_twitch_chat_reads_and_pongs():
    sock = FakeSock([b":a!a@a.tmi.twitch.tv PRIVMSG #chan :hi\r\nPING :tmi.twitch.tv\r\n:b!b@b PRI",
                     b"VMSG #chan :NO WAY\r\n"])
    chat = W.TwitchChat("Chan", connect=lambda *a, **k: sock)
    with pytest.raises(ConnectionError):
        chat.run()
    assert [m.text for m in chat.drain()] == ["hi", "NO WAY"]
    assert "JOIN #chan" in sock.sent[0] and any("PONG" in s for s in sock.sent)


def test_hype_detector_warmup_and_spike():
    det = W.HypeDetector({"baseline_window_seconds": 300, "min_warmup_seconds": 60, "sigma_threshold": 2.0,
                          "reaction_patterns": PATTERNS})
    calm = [W.ChatMessage(f"u{i % 20}", "where are we going") for i in range(12)]
    ticks = [det.update(calm) for _ in range(45)]
    assert not any(t.above for t in ticks)                       # régime calme : jamais au-dessus du seuil
    assert not ticks[10].warmed_up and ticks[40].warmed_up
    spike = [W.ChatMessage(f"new{i}", "NO WAY 💀") for i in range(70)]
    hot = det.update(spike)
    assert hot.above and hot.sigma > 2 and hot.ratio_to_mean > 4 and hot.reaction_ratio == 1.0
    assert hot.new_chatters == 70 and det.update(spike).new_chatters == 0


# ---------------------------------------------------------------- buffer
def make_segments(directory: Path, indices, seconds=2):
    directory.mkdir(parents=True, exist_ok=True)
    for i in indices:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=s=160x90:r=10:d={seconds}", "-f", "lavfi", "-i", f"sine=f={300 + 50 * i}:d={seconds}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", "-f", "mpegts",
                        str(directory / f"seg_{i:05d}.ts")], check=True)


def test_recorder_prune_and_command(tmp_path):
    for i in range(12):
        (tmp_path / f"seg_{i:05d}.ts").write_bytes(b"x")
    calls = []

    class P:
        stdout = None

        def poll(self):
            return None

    rec = W.BufferRecorder("https://twitch.tv/x", tmp_path, keep=8, popen=lambda cmd, **k: calls.append(cmd) or P())
    assert rec.prune(locked={1, 2}) == [0, 3]                     # garde 4..11 (+ verrous 1, 2)
    assert [i for i, _ in W.list_segments(tmp_path)] == [1, 2, *range(4, 12)]
    if shutil.which("ffmpeg"):
        rec.start()
        assert calls[0][1:4] == ["-m", "streamlink", "--stdout"] and "12" in calls[1]   # numérotation reprise
        assert rec.alive()


@needs_ffmpeg
def test_assemble_segments(tmp_path):
    make_segments(tmp_path / "buf", [0, 1, 2])
    out = W.assemble([p for _, p in W.list_segments(tmp_path / "buf")], tmp_path / "inbox" / "raw.mp4")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(out)], capture_output=True, text=True, check=True).stdout)
    assert out.is_file() and dur == pytest.approx(6.0, abs=0.5)


# ---------------------------------------------------------------- Watcher
class FakeRecorder(W.BufferRecorder):
    def __init__(self, directory):
        super().__init__("u", directory, keep=8, segment_seconds=60)
        self.up, self.starts = True, 0

    def start(self):
        self.starts += 1
        self.started_at = 0.0

    def alive(self):
        return self.up

    def stop(self):
        pass


class FakeChat(W.ChatSource):
    name = "fake"

    def start(self):
        pass

    def push(self, msgs):
        for m in msgs:
            self.queue.put(m)


@pytest.fixture
def watcher(project, tmp_path):
    cfg = load_config(project / "config.yaml")
    buf = tmp_path / "buf"
    buf.mkdir()
    for i in range(10):
        (buf / f"seg_{i:05d}.ts").write_bytes(b"x")
    clock = [0.0]
    assembled = []

    def fake_assemble(segs, out):
        assembled.append(([p.name for p in segs], out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return out

    rec = FakeRecorder(buf)
    chat = FakeChat()
    w = W.Watcher(cfg, W.LiveTarget("twitch", "chan", "https://twitch.tv/chan"), recorder=rec, chat=chat,
                  clock=lambda: clock[0], sleep=lambda s: None, assembler=fake_assemble,
                  now=lambda: datetime(2026, 11, 22, 21, 30, tzinfo=timezone.utc))
    w.start()
    return w, rec, chat, clock, assembled, buf


def calm_ticks(w, chat, clock, n):
    for _ in range(n):
        chat.push([W.ChatMessage(f"u{j}", "chill drive") for j in range(10)])
        clock[0] += 2
        w.tick()


def test_watcher_spike_capture_and_cooldown(watcher):
    w, _rec, chat, clock, assembled, buf = watcher
    calm_ticks(w, chat, clock, 40)
    assert not w.state.captures
    chat.push([W.ChatMessage(f"n{j}", "CLIP THAT 💀") for j in range(80)])
    clock[0] += 2
    w.tick()
    cap = w.state.captures[0]
    assert (cap.first, cap.last) == (2, 11) and w.state.status == "rec"      # 8 min passées (2..9) + 2 à venir
    assert cap.file.name == f"raw_{datetime(2026, 11, 22, 21, 30, tzinfo=timezone.utc).astimezone():%Y%m%d_%H%M%S}.mp4"
    # nouveau pic pendant l'enregistrement : ignoré
    chat.push([W.ChatMessage(f"m{j}", "NO WAY") for j in range(90)])
    clock[0] += 2
    w.tick()
    assert len(w.state.captures) == 1
    # les segments 10, 11 puis 12 arrivent : la capture est assemblée quand le 11 est fermé
    for i in (10, 11, 12):
        (buf / f"seg_{i:05d}.ts").write_bytes(b"x")
    calm_ticks(w, chat, clock, 1)
    assert cap.status == "ready" and assembled[0][0] == [f"seg_{i:05d}.ts" for i in range(2, 12)]
    assert w.state.status == "live"
    assert not (buf / "seg_00002.ts").exists()                              # verrou levé -> purgé
    assert any("capture écrite" in e[1] for e in w.state.events)


def test_watcher_reconnects_then_ends(watcher):
    w, rec, _chat, clock, _, _ = watcher
    rec.up = False
    waits = []
    for _ in range(3):
        clock[0] += 0.1
        w.tick()                                    # annonce la coupure
        waits.append(w._retry_at - clock[0])
        clock[0] = w._retry_at
        w.tick()                                    # relance
    assert [round(x, 1) for x in waits] == [5.0, 10.0, 20.0] and rec.starts == 4
    rec.up = True
    clock[0] += 1
    w.tick()
    assert w.state.status == "live" and any("reconnecté" in e[1] for e in w.state.events)
    rec.up = False
    clock[0] += 1
    w.tick()
    assert w.state.status == "ended"


def test_manual_capture_key_and_render(watcher):
    w, rec, chat, clock, _, _ = watcher
    calm_ticks(w, chat, clock, 3)
    w.handle_key("c")
    assert len(w.state.captures) == 1 and w.state.captures[0].reason == "manuel"
    w.handle_key("c")
    assert len(w.state.captures) == 1 and "cooldown" in w.state.events[0][1]
    w.handle_key("p")
    assert "aucune capture prête" in w.state.events[0][1]
    from rich.console import Console

    console = Console(record=True, width=140)
    console.print(W.render_state(w.state, rec))
    text = console.export_text()
    for part in ("REC", "Buffer roulant", "Score de hype", "Chat", "Événements", "Captures", "chill drive"):
        assert part in text


def test_sparkline():
    assert W.sparkline([0, 5, 10], 10, 0) == "▁▄█"


def test_cli_watch_disabled_by_default(project):
    from typer.testing import CliRunner

    import clipgod

    res = CliRunner().invoke(clipgod.app, ["--config", str(project / "config.yaml"), "watch", "https://twitch.tv/x"])
    assert res.exit_code == 1 and "désactivé" in res.output


@needs_ffmpeg
def test_real_pipeline_with_simulated_live(project, tmp_path):
    """Vrai tube vers ffmpeg -f segment ; seul streamlink est remplacé par un « live » ffmpeg de 12 s."""
    import queue
    import subprocess as sp

    def popen(cmd, **kw):
        if "streamlink" in cmd:
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=10",
                   "-f", "lavfi", "-i", "sine=f=440", "-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-g", "10",
                   "-c:a", "aac", "-f", "mpegts", "pipe:1"]
        return sp.Popen(cmd, **kw)

    cfg = load_config(project / "config.yaml")
    cfg["watch"].update({"segment_seconds": 1, "buffer_minutes": 4 / 60, "post_trigger_minutes": 2 / 60,
                         "chat_poll_seconds": 0.5})
    cfg["watch"]["reconnect"]["max_attempts"] = 0
    rec = W.BufferRecorder("sim://live", tmp_path / "buf", segment_seconds=1, keep=4, popen=popen)
    w = W.Watcher(cfg, W.LiveTarget("twitch", "sim", "sim://live"), recorder=rec, chat=None)
    keys: queue.Queue = queue.Queue()
    seen_max = []

    def render(state):
        segs = W.list_segments(rec.dir)
        seen_max.append(len(segs))
        if len(segs) >= 4 and not state.captures:
            keys.put("c")

    w.run(render=render, keys=keys)                 # s'arrête à la fin du « live » (aucune reconnexion permise)
    cap = w.state.captures[0]
    assert cap.status == "ready" and cap.file.is_file(), [e[1] for e in w.state.events]
    dur = float(sp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(cap.file)],
                       capture_output=True, text=True, check=True).stdout)
    assert 3 <= dur <= 8                             # ~4 s de buffer + ~2 s après le déclenchement
    assert max(seen_max) <= 4 + 1 + 3                # purge active (buffer + segment en cours + verrous)
    assert w.state.status == "ended"
