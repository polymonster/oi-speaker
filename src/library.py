"""Music on a local disk: an index of the tracks under the configured folders, searchable by artist,
album, track, genre and folder, and resolved to the file paths mpv plays. Tags are read with
mutagen once and cached, so a restart only re-reads what changed on the disk."""
import difflib
import hashlib
import json
import os
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

LIBRARY_INDEX_PATH = Path(".library.json")
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".flac", ".ogg", ".opus", ".wav", ".aac", ".wma", ".aif", ".aiff")
RESCAN_INTERVAL = 3600.0  # seconds before a search walks the disk again for added or removed files
PLAY_LIMIT = 500  # tracks queued at most — a whole-library shuffle does not need all of them
FUZZY_CUTOFF = 0.75  # how close a misheard artist or album must be to count as a match
# how many of each kind a search returns — enough to choose from, few enough to stay cheap in context
RESULT_LIMITS = {"artists": 5, "albums": 8, "tracks": 10, "genres": 5, "folders": 5}

_TRACK_PREFIX = re.compile(r"^\s*[A-Da-d]?\d{1,3}(?:\s*[-.)_]\s*|\s+)")  # "04 - ", "03.", "A1 - "
_ID_PREFIX = {"track": "t", "artist": "r", "album": "a", "genre": "g", "folder": "f"}
_CATALOGUE_SUFFIX = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]\s*$")  # "Album (CAT 001)"
_GENRE_SPLIT = re.compile(r"\s*[,;/]\s*")


@dataclass
class _Index:
    """One complete scan, swapped in whole so a search never sees a half built index."""
    tracks: list[dict] = field(default_factory=list)
    groups: dict[str, dict] = field(default_factory=dict)  # id -> {kind, name, artist?, year?, tracks: [indices]}
    track_ids: dict[str, int] = field(default_factory=dict)
    paths: dict[str, int] = field(default_factory=dict)
    roots: tuple[str, ...] = ()
    scanned_at: float = 0.0


class Library:
    """The tracks under `roots`. Scans on a background thread; searches run against the last
    complete scan, and kick off a fresh one when a folder appears or vanishes (a USB drive plugged
    in or pulled) or the last scan is older than RESCAN_INTERVAL."""

    def __init__(self, roots: list[str], log=print, path: Path = LIBRARY_INDEX_PATH):
        self._log = log
        self._path = path
        self._roots = _clean_roots(roots)
        self._index = _Index()
        self._scanning = threading.Lock()
        self._progress = 0

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    def start(self):
        """Load the cached index, so search works at once, then rescan in the background."""
        cached = self._load_cache()
        present = tuple(r for r in self._roots if os.path.isdir(r))
        tracks = [t for t in cached.values() if _under(t["path"], present)]
        self._index = _build(tracks, present, scanned_at=0.0)  # 0: stale, so the rescan still runs
        if self._roots:
            self.rescan()

    def set_roots(self, roots: list[str]):
        roots = _clean_roots(roots)
        if roots != self._roots:
            self._roots = roots
            self.rescan()

    def rescan(self):
        threading.Thread(target=self._scan, daemon=True).start()

    def status(self) -> dict:
        index = self._index
        return {
            "roots": self._roots,
            "missing": [r for r in self._roots if not os.path.isdir(r)],
            "tracks": len(index.tracks),
            "scanning": self._scanning.locked(),
            "progress": self._progress,
            "scanned_at": index.scanned_at,
        }

    def _load_cache(self) -> dict[str, dict]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return {t["path"]: t for t in data.get("tracks", [])}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._log(f"library: could not read {self._path}, rebuilding: {e}")
            return {}

    def _scan(self):
        if not self._scanning.acquire(blocking=False):
            return  # one at a time; the running scan will do
        try:
            started = time.time()
            roots = tuple(self._roots)
            present = tuple(r for r in roots if os.path.isdir(r))
            for root in roots:
                if root not in present:
                    self._log(f"library: {root} is not there, skipping it")
            # the cache keeps tracks from every root, so a drive that is unplugged today is not
            # re-read from scratch when it comes back
            cache = self._load_cache()
            tracks, read = [], 0
            self._progress = 0
            for root in present:
                for path, size, mtime in _walk(root):
                    track = cache.get(path)
                    if track is None or track.get("size") != size or track.get("mtime") != mtime:
                        track = _read_track(path, root, size, mtime)
                        cache[path] = track
                        read += 1
                        if read % 1000 == 0:
                            self._log(f"library: read tags of {read} new files...")
                    tracks.append(track)
                    self._progress = len(tracks)
            # gone from a present root means deleted; under an absent root, kept for its return
            cache = {p: t for p, t in cache.items() if not _under(p, present)}
            cache.update((t["path"], t) for t in tracks)
            self._index = _build(tracks, present, scanned_at=time.time())
            try:
                saved = [{k: v for k, v in t.items() if k != "_key"} for t in cache.values()]
                self._path.write_text(json.dumps({"tracks": saved}), encoding="utf-8")
            except OSError as e:
                self._log(f"library: could not save {self._path}: {e}")
            self._log(f"library: {len(tracks)} tracks in {', '.join(present) or 'nothing'} "
                      f"({read} read, {time.time() - started:.1f}s)")
        except Exception as e:
            self._log(f"library: scan failed: {e}")
        finally:
            self._scanning.release()

    def _check_fresh(self):
        index = self._index
        present = tuple(r for r in self._roots if os.path.isdir(r))
        if present != index.roots or time.time() - index.scanned_at > RESCAN_INTERVAL:
            self.rescan()

    def search(self, query: str) -> dict:
        """Artists, albums, tracks, genres and folders matching `query`, best first. Every word of the
        query has to start a word of the name; failing that, near misses on artist and album names
        catch a misheard word."""
        self._check_fresh()
        index = self._index
        words = _norm(query).split()
        if not words:
            return {}
        results: dict[str, list] = {kind: [] for kind in RESULT_LIMITS}
        for gid, group in index.groups.items():
            if _matches(words, group["key"]):
                results[group["kind"] + "s"].append(gid)
        for i, track in enumerate(index.tracks):
            if _matches(words, track["_key"]):
                results["tracks"].append(i)
        if not any(results.values()):
            self._fuzzy(index, " ".join(words), results)
        out = {}
        query_norm = " ".join(words)
        for kind, hits in results.items():
            if not hits:
                continue
            if kind == "tracks":
                hits.sort(key=lambda i: (_norm(index.tracks[i]["title"]) != query_norm, len(index.tracks[i]["_key"])))
                rows = [_track_row(index.tracks[i]) for i in hits[:RESULT_LIMITS[kind]]]
            else:
                if kind == "folders":
                    # copies of one folder in several places look the same, and one is enough
                    copies = {(index.groups[g]["norm"], len(index.groups[g]["tracks"])): g for g in reversed(hits)}
                    keep = set(copies.values())
                    hits = [g for g in hits if g in keep]
                hits.sort(key=lambda g: (index.groups[g]["norm"] != query_norm, -len(index.groups[g]["tracks"])))
                rows = [_group_row(g, index.groups[g]) for g in hits[:RESULT_LIMITS[kind]]]
            out[kind] = rows
            if len(hits) > len(rows):
                out[f"more_{kind}"] = len(hits) - len(rows)
        return out

    def _fuzzy(self, index: _Index, query: str, results: dict[str, list]):
        by_name: dict[str, list[str]] = {}
        for gid, group in index.groups.items():
            if group["kind"] in ("artist", "album"):
                by_name.setdefault(group["norm"], []).append(gid)
        for name in difflib.get_close_matches(query, by_name, n=5, cutoff=FUZZY_CUTOFF):
            for gid in by_name[name]:
                results[index.groups[gid]["kind"] + "s"].append(gid)

    def resolve(self, ids: list[str], shuffle: bool = False) -> tuple[list[str], list[str]]:
        """File paths for track and group ids, in album order unless shuffled — or the whole library,
        shuffled, when `ids` is empty. Returns (paths, unknown ids)."""
        index = self._index
        picked, unknown, seen = [], [], set()
        if not ids:
            picked, shuffle = list(range(len(index.tracks))), True
        for gid in ids:
            if gid in index.groups:
                members = index.groups[gid]["tracks"]
            elif gid in index.track_ids:
                members = [index.track_ids[gid]]
            else:
                unknown.append(gid)
                continue
            for i in members:
                if i not in seen:
                    seen.add(i)
                    picked.append(i)
        if shuffle:
            random.shuffle(picked)
        return [index.tracks[i]["path"] for i in picked[:PLAY_LIMIT]], unknown

    def title(self, ids: list[str]) -> str:
        """A name for what `ids` play, for the play history: 'Nevermind - Nirvana', 'Burial', ..."""
        index = self._index
        names = []
        for gid in ids:
            if gid in index.groups:
                group = index.groups[gid]
                names.append(f"{group['name']} - {group['artist']}" if group.get("artist") else group["name"])
            elif gid in index.track_ids:
                track = index.tracks[index.track_ids[gid]]
                names.append(f"{track['title']} - {track['artist']}" if track["artist"] else track["title"])
        return ", ".join(names) or "the whole library, shuffled"

    def describe(self, path: str) -> dict | None:
        """The tags of a file in the index, for saying what is playing."""
        i = self._index.paths.get(path)
        return _track_row(self._index.tracks[i]) if i is not None else None


def _clean_roots(roots) -> list[str]:
    if isinstance(roots, str):
        roots = [roots]
    out = []
    for root in roots or []:
        root = str(root).strip()
        if root:
            out.append(os.path.normpath(os.path.expanduser(root)))
    return out


def _under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def _walk(root: str):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not d.startswith("$"))
        for name in sorted(files):
            if name.startswith(".") or not name.lower().endswith(AUDIO_EXTENSIONS):
                continue  # ._ files are macOS resource forks, not audio
            path = os.path.join(folder, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            yield path, stat.st_size, int(stat.st_mtime)


def _first(tags, key: str) -> str:
    try:
        value = tags.get(key)
    except (KeyError, ValueError):
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _number(text: str) -> int:
    match = re.match(r"\d+", text or "")
    return int(match.group()) if match else 0


def _read_track(path: str, root: str, size: int, mtime: int) -> dict:
    """Tags from the file, with gaps filled from the folder layout ("Artist - Album/04 - Title")."""
    tags, duration = {}, 0
    try:
        import mutagen
        audio = mutagen.File(path, easy=True)
        if audio is not None:
            tags = audio.tags or {}
            duration = int(getattr(audio.info, "length", 0) or 0)
    except Exception:
        pass  # unreadable tags — the path still says plenty
    guess = _guess_from_path(path, root)
    return {
        "path": path,
        "root": root,
        "size": size,
        "mtime": mtime,
        "title": _first(tags, "title") or guess["title"],
        "artist": _first(tags, "artist") or guess["artist"],
        "albumartist": _first(tags, "albumartist"),
        "album": _first(tags, "album") or guess["album"],
        "genre": _first(tags, "genre"),
        "year": _number(_first(tags, "date")),
        "disc": _number(_first(tags, "discnumber")),
        "track": _number(_first(tags, "tracknumber")) or _number(os.path.basename(path)),
        "duration": duration,
    }


def _guess_from_path(path: str, root: str) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0]
    title = _TRACK_PREFIX.sub("", stem).strip() or stem
    artist = ""
    if " - " in title:
        artist, title = (part.strip() for part in title.split(" - ", 1))
    folder = os.path.dirname(path)
    album = "" if os.path.normpath(folder) == os.path.normpath(root) else os.path.basename(folder)
    album = _CATALOGUE_SUFFIX.sub("", album).strip()
    if " - " in album:
        folder_artist, album = (part.strip() for part in album.split(" - ", 1))
        artist = artist or folder_artist
    return {"title": title, "artist": artist, "album": album}


def _norm(text: str) -> str:
    """Lower case, accents and punctuation gone — 'Röyksopp' is said and transcribed 'royksopp'."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower().replace("&", " and ")
    text = re.sub(r"^(.+),\s*the$", r"the \1", text.strip())  # "Cure, The" is filed as "The Cure"
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _matches(words: list[str], key: str) -> bool:
    return all(f" {w}" in key for w in words)


def _gid(kind: str, name: str) -> str:
    return _ID_PREFIX[kind] + hashlib.sha1(f"{kind}:{name}".encode("utf-8")).hexdigest()[:7]


def _build(tracks: list[dict], roots: tuple[str, ...], scanned_at: float) -> _Index:
    """Group the tracks into artists, albums, genres and folders, each with a stable id (a hash of
    what it is, so an id the llm saw before a rescan still plays the same thing after it).

    The same album is often on the disk more than once, copied into several folders. Only the first
    copy of a track counts toward its artist, album and genre, and only it is searchable on its own,
    so nothing plays twice or shows up three times — a folder still plays everything inside it."""
    tracks = sorted(tracks, key=lambda t: (t["path"].lower()))
    groups: dict[str, dict] = {}
    seen: set[tuple] = set()

    def add(kind: str, key: str, name: str, i: int, **extra):
        gid = _gid(kind, key)
        group = groups.get(gid)
        if group is None:
            group = groups[gid] = {"kind": kind, "name": name, "norm": _norm(name), "tracks": [], **extra}
        group["tracks"].append(i)

    for i, track in enumerate(tracks):
        album_artist = track["albumartist"] or track["artist"]
        copy = (_norm(album_artist), _norm(track["album"]), _norm(track["title"]), track["disc"], track["track"],
                track["duration"])
        duplicate = copy in seen
        seen.add(copy)
        # a duplicate has an empty key, which no search matches
        track["_key"] = "" if duplicate else " " + _norm(f"{track['title']} {track['artist']} {album_artist}")
        if not duplicate:
            for artist in {track["artist"], track["albumartist"]} - {"", "Various", "Various Artists"}:
                add("artist", _norm(artist), artist, i)
            if track["album"]:
                add("album", f"{_norm(album_artist)}|{_norm(track['album'])}", track["album"], i,
                    artist=album_artist, year=track["year"])
            for genre in _GENRE_SPLIT.split(track["genre"]):
                if genre:
                    add("genre", _norm(genre), genre, i)
        # every folder between the root and the file, so 'the wedding folder' plays all beneath it
        folder = os.path.dirname(track["path"])
        root = track["root"]
        while _under(folder, (root,)) and folder != root:
            add("folder", folder, os.path.basename(folder), i)
            folder = os.path.dirname(folder)

    for group in groups.values():
        names = group["name"] + (f" {group['artist']}" if group["kind"] == "album" else "")
        group["key"] = " " + _norm(names)
        # album order: disc, track number, file name
        group["tracks"].sort(key=lambda i: (tracks[i]["albumartist"] or tracks[i]["artist"], tracks[i]["album"],
                                            tracks[i]["disc"], tracks[i]["track"], tracks[i]["path"]))
        if group["kind"] == "folder":
            group["tracks"].sort(key=lambda i: tracks[i]["path"].lower())

    return _Index(tracks=tracks, groups=groups,
                  track_ids={_gid("track", t["path"]): i for i, t in enumerate(tracks)},
                  paths={t["path"]: i for i, t in enumerate(tracks)},
                  roots=roots, scanned_at=scanned_at)


def _track_row(track: dict) -> dict:
    row = {"id": _gid("track", track["path"]), "title": track["title"], "artist": track["artist"]}
    if track["album"]:
        row["album"] = track["album"]
    if track["duration"]:
        row["duration"] = track["duration"]
    return row


def _group_row(gid: str, group: dict) -> dict:
    row = {"id": gid, "name": group["name"], "tracks": len(group["tracks"])}
    if group["kind"] == "album":
        row["artist"] = group["artist"]
        if group["year"]:
            row["year"] = group["year"]
    return row
