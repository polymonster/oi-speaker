"""Clock alarms: the saved list, iOS style (a time, a label, the days it repeats on, a sound and
whether it can snooze), and the scheduler that fires them. What a firing alarm sounds like is the
speaker's business — the clock only calls back with the alarm that is due."""
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

ALARMS_PATH = Path("alarms.json")  # this box's alarms, not synced between speakers
ALARM_SOUND_DIR = Path("sounds/alarms")
SOUND_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
DEFAULT_SOUND = "sounds/alarms/midnight.wav"
SNOOZE_MINUTES = 9

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # datetime.weekday() order
# what the llm (or a person) might call a set of days
DAY_GROUPS = {
    "weekdays": [0, 1, 2, 3, 4],
    "weekends": [5, 6],
    "daily": [0, 1, 2, 3, 4, 5, 6],
    "everyday": [0, 1, 2, 3, 4, 5, 6],
    "every day": [0, 1, 2, 3, 4, 5, 6],
}


class AlarmClock:
    """The saved alarms and the timer sound, and a thread that calls `on_fire(alarm)` at the start
    of each minute an enabled alarm is due. A one-off alarm (no days) turns itself off once fired."""

    def __init__(self, on_fire, log=print, path: Path = ALARMS_PATH):
        self._on_fire = on_fire
        self._log = log
        self._path = path
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._last_minute: str | None = None
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        except (OSError, ValueError) as e:
            self._log(f"alarms: could not read {self._path}, starting empty: {e}")
            data = {}
        return normalise(data)

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def data(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))  # a copy the caller can't mutate us through

    def timer_sound(self) -> str:
        with self._lock:
            return self._data["timer_sound"]

    def replace(self, data: dict) -> dict:
        """Swap in a whole edited list, as the alarms page saves it. Returns what was stored."""
        with self._lock:
            self._data = normalise(data)
            self._save()
            stored = json.loads(json.dumps(self._data))
        self._changed.set()
        return stored

    def add(self, time: str, label: str = "", days: list[int] | None = None, sound: str = "") -> dict:
        alarm = normalise_alarm({"time": time, "label": label, "days": days or [], "sound": sound})
        with self._lock:
            self._data["alarms"].append(alarm)
            self._data["alarms"].sort(key=lambda a: a["time"])
            self._save()
        self._changed.set()
        return alarm

    def remove(self, match: str) -> list[dict]:
        """Delete the alarms whose time or label is `match`. Returns the ones removed."""
        key = match.strip().lower()
        time = parse_time(match)
        with self._lock:
            gone = [a for a in self._data["alarms"] if a["label"].lower() == key or (time and a["time"] == time)]
            self._data["alarms"] = [a for a in self._data["alarms"] if a not in gone]
            if gone:
                self._save()
        return gone

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            now = datetime.now()
            minute = now.strftime("%Y-%m-%d %H:%M")
            # once a minute, so a save landing mid-minute can't fire the same alarm twice
            if minute != self._last_minute:
                self._last_minute = minute
                for alarm in self._due(now):
                    self._log(f"alarm fired: {describe(alarm)}")
                    try:
                        self._on_fire(alarm)
                    except Exception as e:
                        self._log(f"alarm error: {e}")
            # sleep to just past the next minute, or until the list changes
            self._changed.wait(timeout=60.05 - now.second - now.microsecond / 1e6)
            self._changed.clear()

    def _due(self, now: datetime) -> list[dict]:
        hhmm = now.strftime("%H:%M")
        with self._lock:
            due = [a for a in self._data["alarms"]
                   if a["enabled"] and a["time"] == hhmm and (not a["days"] or now.weekday() in a["days"])]
            once = [a for a in due if not a["days"]]
            for alarm in once:
                alarm["enabled"] = False
            if once:
                self._save()
            return [dict(a) for a in due]


def list_sounds() -> list[str]:
    """Every clip in the alarm sounds folder. Read on each call, so dropping a file in is enough."""
    if not ALARM_SOUND_DIR.is_dir():
        return []
    return sorted(p.as_posix() for p in ALARM_SOUND_DIR.iterdir() if p.suffix.lower() in SOUND_EXTENSIONS)


def parse_time(text: str) -> str | None:
    """'7:00', '07:00', '19:30' as 24 hour 'HH:MM', or None."""
    try:
        hour, minute = (int(part) for part in text.strip().split(":"))
    except ValueError:
        return None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return f"{hour:02d}:{minute:02d}"


def parse_days(days) -> list[int]:
    """Day numbers (0 = monday) from numbers, names ('mon', 'Tuesday') or groups ('weekdays')."""
    if isinstance(days, str):
        days = [days]
    out = set()
    for day in days or []:
        if isinstance(day, int) and 0 <= day < 7:
            out.add(day)
            continue
        name = str(day).strip().lower()
        if name in DAY_GROUPS:
            out.update(DAY_GROUPS[name])
        elif name[:3] in DAY_NAMES:
            out.add(DAY_NAMES.index(name[:3]))
    return sorted(out)


def _pick_sound(sound: str | None) -> str:
    """`sound` if it is still there, else the default, else whatever there is — a sound file
    deleted from the folder must not leave an alarm silent."""
    sounds = list_sounds()
    for choice in (sound, DEFAULT_SOUND):
        if choice in sounds:
            return choice
    return sounds[0] if sounds else DEFAULT_SOUND


def normalise_alarm(alarm: dict) -> dict:
    return {
        "id": str(alarm.get("id") or uuid.uuid4().hex[:8]),
        "time": parse_time(str(alarm.get("time", ""))) or "07:00",
        "label": str(alarm.get("label", "")).strip(),
        "days": parse_days(alarm.get("days", [])),
        "sound": _pick_sound(alarm.get("sound")),
        "snooze": bool(alarm.get("snooze", True)),
        "enabled": bool(alarm.get("enabled", True)),
    }


def normalise(data: dict) -> dict:
    return {
        "timer_sound": _pick_sound(data.get("timer_sound")),
        "alarms": sorted((normalise_alarm(a) for a in data.get("alarms", [])), key=lambda a: a["time"]),
    }


def describe_days(days: list[int]) -> str:
    if not days:
        return "once"
    for name, group in DAY_GROUPS.items():
        if days == group:
            return "every day" if len(group) == 7 else name
    return ", ".join(DAY_NAMES[d] for d in days)


def describe(alarm: dict) -> str:
    label = f" '{alarm['label']}'" if alarm["label"] else ""
    state = "" if alarm["enabled"] else " (off)"
    return f"{alarm['time']}{label} {describe_days(alarm['days'])}{state}"
