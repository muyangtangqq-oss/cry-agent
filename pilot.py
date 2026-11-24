"""Local pilot harness for 3-5 volunteer families.

The pilot is intentionally local-first:
  - pseudonymous family IDs, no names
  - audio retention is opt-in per family
  - every prediction can carry parent feedback
  - reports and anonymized exports are generated from SQLite

This does not diagnose or treat anything. It collects evidence for whether
the classifier and multimodal fusion work in real homes.

CLI:
  python pilot.py init --consent --age-days 42 --audio-retention
  python pilot.py run --family fam_xxx --audio clip.wav --postures rooting
  python pilot.py feedback --event 12 --cause hungry
  python pilot.py report
  python pilot.py export --out pilot_export
"""
import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import statistics
import time
from collections import Counter, defaultdict

DEFAULT_ROOT = os.path.expanduser("~/.config/cry-agent/pilot")
CONSENT_VERSION = "pilot-1.0"


def _json_default(value):
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PilotStore:
    def __init__(self, root=None):
        self.root = os.path.expanduser(
            root or os.environ.get("CRY_AGENT_PILOT_DIR") or DEFAULT_ROOT
        )
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "audio"), exist_ok=True)
        self.db_path = os.path.join(self.root, "pilot.db")
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS families (
                family_id TEXT PRIMARY KEY,
                age_days INTEGER NOT NULL,
                consent_version TEXT NOT NULL,
                consented_at REAL NOT NULL,
                audio_retention INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                withdrawn INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                family_id TEXT NOT NULL,
                ts REAL NOT NULL,
                scenario TEXT NOT NULL,
                cause TEXT,
                confidence REAL,
                model TEXT,
                fused_cause TEXT,
                fused_confidence REAL,
                audio_cause TEXT,
                audio_confidence REAL,
                vision_cause TEXT,
                vision_confidence REAL,
                escalated INTEGER NOT NULL DEFAULT 0,
                critical_override INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL,
                audio_path TEXT,
                signals_json TEXT,
                feedback_cause TEXT,
                feedback_correct INTEGER,
                feedback_ts REAL,
                feedback_notes TEXT,
                FOREIGN KEY (family_id) REFERENCES families(family_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_family
                ON events(family_id, ts);
            """)

    # ---- families -----------------------------------------------------------
    def add_family(self, age_days, consent_version=CONSENT_VERSION,
                   audio_retention=False, notes="", family_id=None):
        family_id = family_id or f"fam_{secrets.token_hex(4)}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO families "
                "(family_id, age_days, consent_version, consented_at, "
                " audio_retention, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (family_id, int(age_days), consent_version, time.time(),
                 int(bool(audio_retention)), notes),
            )
        return family_id

    def get_family(self, family_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM families WHERE family_id = ?", (family_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_families(self):
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT f.*, COUNT(e.id) AS events
                FROM families f LEFT JOIN events e ON e.family_id = f.family_id
                GROUP BY f.family_id ORDER BY f.consented_at
            """).fetchall()
        return [dict(r) for r in rows]

    def withdraw(self, family_id, delete_audio=True):
        with self._connect() as conn:
            conn.execute(
                "UPDATE families SET withdrawn = 1 WHERE family_id = ?",
                (family_id,),
            )
            conn.execute(
                "UPDATE events SET audio_path = NULL WHERE family_id = ?",
                (family_id,),
            )
        audio_dir = os.path.join(self.root, "audio", family_id)
        if delete_audio and os.path.isdir(audio_dir):
            shutil.rmtree(audio_dir)

    # ---- events -------------------------------------------------------------
    def log_result(self, family_id, result, audio_path=None, signals=None):
        family = self.get_family(family_id)
        if not family:
            raise ValueError(f"Unknown family: {family_id}")
        if family["withdrawn"]:
            raise ValueError(f"Family {family_id} has withdrawn consent")

        payload = {
            "cause": _attr(result, "cause"),
            "confidence": _attr(result, "confidence"),
            "model": _attr(result, "model"),
            "fused_cause": _attr(result, "fused_cause"),
            "fused_confidence": _attr(result, "fused_confidence"),
            "audio_cause": _attr(result, "audio_cause"),
            "audio_confidence": _attr(result, "audio_confidence"),
            "vision_cause": _attr(result, "vision_cause"),
            "vision_confidence": _attr(result, "vision_confidence"),
            "escalated": int(bool(_attr(result, "escalated", False))),
            "critical_override": int(bool(_attr(result, "critical_override", False))),
            "latency_ms": _attr(result, "latency_ms"),
            "scenario": _attr(result, "scenario", "baby_cry"),
            "signals_json": json.dumps(
                signals if signals is not None else _attr(result, "per_signal", {}),
                default=_json_default,
            ),
        }
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO events (
                    family_id, ts, scenario, cause, confidence, model,
                    fused_cause, fused_confidence, audio_cause, audio_confidence,
                    vision_cause, vision_confidence, escalated,
                    critical_override, latency_ms, signals_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                family_id, time.time(), payload["scenario"], payload["cause"],
                payload["confidence"], payload["model"], payload["fused_cause"],
                payload["fused_confidence"], payload["audio_cause"],
                payload["audio_confidence"], payload["vision_cause"],
                payload["vision_confidence"], payload["escalated"],
                payload["critical_override"], payload["latency_ms"],
                payload["signals_json"],
            ))
            event_id = cur.lastrowid

        if audio_path and family["audio_retention"]:
            ext = os.path.splitext(audio_path)[1] or ".wav"
            dest_dir = os.path.join(self.root, "audio", family_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"{event_id:06d}{ext}")
            shutil.copy2(audio_path, dest)
            rel = os.path.relpath(dest, self.root)
            with self._connect() as conn:
                conn.execute(
                    "UPDATE events SET audio_path = ? WHERE id = ?",
                    (rel, event_id),
                )
        return event_id

    def record_feedback(self, event_id, actual_cause=None, correct=None,
                        notes=""):
        with self._connect() as conn:
            event = conn.execute(
                "SELECT cause FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            if not event:
                raise ValueError(f"Unknown event: {event_id}")
            if correct is None and actual_cause is not None:
                correct = event["cause"] == actual_cause
            if correct is None:
                raise ValueError("Provide either actual_cause or correct")
            conn.execute("""
                UPDATE events SET feedback_cause = ?, feedback_correct = ?,
                    feedback_ts = ?, feedback_notes = ? WHERE id = ?
            """, (actual_cause, int(bool(correct)), time.time(), notes, event_id))

    def get_event(self, event_id):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return dict(row) if row else None

    # ---- reporting ----------------------------------------------------------
    def report(self, family_id=None):
        query = "SELECT * FROM events"
        params = ()
        if family_id:
            query += " WHERE family_id = ?"
            params = (family_id,)
        query += " ORDER BY ts"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
            families = [dict(r) for r in conn.execute(
                "SELECT * FROM families ORDER BY consented_at"
            ).fetchall()]

        feedback = [r for r in rows if r["feedback_correct"] is not None]
        correct = sum(1 for r in feedback if r["feedback_correct"])
        latencies = sorted(
            r["latency_ms"] for r in rows if r["latency_ms"] is not None
        )
        per_cause = defaultdict(lambda: {"total": 0, "correct": 0})
        for row in feedback:
            label = row["feedback_cause"] or row["cause"] or "unknown"
            per_cause[label]["total"] += 1
            per_cause[label]["correct"] += int(bool(row["feedback_correct"]))

        summary = {
            "families": len(families),
            "active_families": sum(1 for f in families if not f["withdrawn"]),
            "events": len(rows),
            "feedback_events": len(feedback),
            "feedback_coverage": round(len(feedback) / max(len(rows), 1), 3),
            "accuracy_on_feedback": (
                round(correct / len(feedback), 3) if feedback else None
            ),
            "model_usage": dict(Counter(r["model"] for r in rows)),
            "escalation_rate": round(
                sum(1 for r in rows if r["escalated"]) / max(len(rows), 1), 3
            ),
            "override_rate": round(
                sum(1 for r in rows if r["critical_override"])
                / max(len(rows), 1), 3
            ),
            "latency_ms": {
                "p50": round(statistics.median(latencies), 1) if latencies else None,
                "p95": round(latencies[int(len(latencies) * 0.95) - 1], 1)
                if latencies else None,
            },
            "audio_retained": sum(1 for r in rows if r["audio_path"]),
            "per_cause_feedback": {
                cause: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"], 3),
                }
                for cause, stats in sorted(per_cause.items())
            },
        }

        if not family_id:
            by_family = {}
            for fam in families:
                fam_rows = [r for r in rows if r["family_id"] == fam["family_id"]]
                fam_feedback = [
                    r for r in fam_rows if r["feedback_correct"] is not None
                ]
                by_family[fam["family_id"]] = {
                    "age_days": fam["age_days"],
                    "withdrawn": bool(fam["withdrawn"]),
                    "events": len(fam_rows),
                    "feedback": len(fam_feedback),
                    "accuracy": round(
                        sum(1 for r in fam_feedback if r["feedback_correct"])
                        / max(len(fam_feedback), 1), 3
                    ) if fam_feedback else None,
                }
            summary["by_family"] = by_family
        return summary

    # ---- export -------------------------------------------------------------
    def _export_salt(self):
        path = os.path.join(self.root, ".export_salt")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(secrets.token_hex(32))
        with open(path) as f:
            return f.read().strip().encode()

    def export(self, out_dir, family_id=None, include_audio=False,
               include_notes=False):
        out_dir = os.path.expanduser(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        salt = self._export_salt()

        def fam_hash(value):
            return hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()[:16]

        query = """
            SELECT e.*, f.age_days, f.consent_version, f.withdrawn
            FROM events e JOIN families f ON f.family_id = e.family_id
        """
        params = ()
        if family_id:
            query += " WHERE e.family_id = ?"
            params = (family_id,)
        query += " ORDER BY e.ts"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]

        fields = [
            "event_id", "family_hash", "timestamp", "scenario", "cause",
            "confidence", "model", "fused_cause", "fused_confidence",
            "audio_cause", "audio_confidence", "vision_cause",
            "vision_confidence", "escalated", "critical_override",
            "latency_ms", "feedback_cause", "feedback_correct",
            "audio_retained", "age_days", "consent_version",
        ]
        if include_notes:
            fields.append("feedback_notes")
        csv_path = os.path.join(out_dir, "events.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            audio_count = 0
            for row in rows:
                retained = bool(row["audio_path"])
                if include_audio and retained:
                    src = os.path.join(self.root, row["audio_path"])
                    if os.path.exists(src):
                        dest_dir = os.path.join(out_dir, "audio")
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy2(
                            src,
                            os.path.join(dest_dir, os.path.basename(src)),
                        )
                        audio_count += 1
                record = {
                    "event_id": row["id"],
                    "family_hash": fam_hash(row["family_id"]),
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(row["ts"])
                    ),
                    "scenario": row["scenario"],
                    "cause": row["cause"],
                    "confidence": row["confidence"],
                    "model": row["model"],
                    "fused_cause": row["fused_cause"],
                    "fused_confidence": row["fused_confidence"],
                    "audio_cause": row["audio_cause"],
                    "audio_confidence": row["audio_confidence"],
                    "vision_cause": row["vision_cause"],
                    "vision_confidence": row["vision_confidence"],
                    "escalated": row["escalated"],
                    "critical_override": row["critical_override"],
                    "latency_ms": row["latency_ms"],
                    "feedback_cause": row["feedback_cause"],
                    "feedback_correct": row["feedback_correct"],
                    "audio_retained": retained,
                    "age_days": row["age_days"],
                    "consent_version": row["consent_version"],
                }
                if include_notes:
                    record["feedback_notes"] = row["feedback_notes"]
                writer.writerow(record)

        manifest = {
            "schema": "cry-agent-pilot-export-v1",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": len(rows),
            "audio_files": audio_count,
            "families": len({r["family_id"] for r in rows}),
            "include_audio": bool(include_audio),
            "include_notes": bool(include_notes),
            "notes": "Family IDs are HMAC-pseudonymized. No names are stored.",
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def export_training_set(self, out_dir, family_id=None):
        """Copy only labeled, audio-retained events into class folders.

        This is the bridge from household feedback to offline retraining.
        """
        out_dir = os.path.expanduser(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        query = """
            SELECT id, family_id, audio_path, feedback_cause
            FROM events
            WHERE feedback_cause IS NOT NULL AND audio_path IS NOT NULL
        """
        params = ()
        if family_id:
            query += " AND family_id = ?"
            params = (family_id,)
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]

        counts = Counter()
        for row in rows:
            src = os.path.join(self.root, row["audio_path"])
            if not os.path.exists(src):
                continue
            label = row["feedback_cause"]
            label_dir = os.path.join(out_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            ext = os.path.splitext(src)[1] or ".wav"
            shutil.copy2(
                src, os.path.join(label_dir, f"pilot_{row['id']:06d}{ext}")
            )
            counts[label] += 1

        manifest = {
            "schema": "cry-agent-pilot-training-v1",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "clips": sum(counts.values()),
            "per_class": dict(sorted(counts.items())),
            "note": "Only events with parent-confirmed cause and opt-in audio.",
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest


def _pairs(items):
    values = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"Expected key=value, got: {item}")
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def _print_report(report):
    print(f"Families:           {report['active_families']}/{report['families']} active")
    print(f"Events:             {report['events']}")
    print(f"Feedback coverage:  {report['feedback_coverage']:.0%} "
          f"({report['feedback_events']} events)")
    if report["accuracy_on_feedback"] is None:
        print("Accuracy:           n/a")
    else:
        print(f"Accuracy:           {report['accuracy_on_feedback']:.1%}")
    print(f"Escalation rate:    {report['escalation_rate']:.0%}")
    print(f"Override rate:      {report['override_rate']:.0%}")
    print(f"Latency p50/p95:    {report['latency_ms']['p50']} / "
          f"{report['latency_ms']['p95']} ms")
    print(f"Audio retained:     {report['audio_retained']}")
    if report["per_cause_feedback"]:
        print("\nFeedback by cause:")
        for cause, stats in report["per_cause_feedback"].items():
            print(f"  {cause:15s} {stats['correct']:>3d}/{stats['total']:<3d} "
                  f"{stats['accuracy']:.0%}")
    if report.get("by_family"):
        print("\nBy family:")
        for family_id, stats in report["by_family"].items():
            acc = "n/a" if stats["accuracy"] is None else f"{stats['accuracy']:.0%}"
            print(f"  {family_id:16s} age={stats['age_days']:>4d}d "
                  f"events={stats['events']:>3d} feedback={stats['feedback']:>3d} "
                  f"acc={acc}")


def main():
    parser = argparse.ArgumentParser(description="cry-agent pilot harness")
    parser.add_argument("--root", help="pilot data directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="enroll one family")
    p_init.add_argument("--age-days", type=int, required=True)
    p_init.add_argument("--consent", action="store_true", required=True)
    p_init.add_argument("--consent-version", default=CONSENT_VERSION)
    p_init.add_argument("--audio-retention", action="store_true")
    p_init.add_argument("--notes", default="")
    p_init.add_argument("--family-id")

    p_run = sub.add_parser("run", help="run the pipeline once and log it")
    p_run.add_argument("--family", required=True)
    p_run.add_argument("--scenario", default="baby_cry")
    p_run.add_argument("--audio")
    p_run.add_argument("--postures", default="")
    p_run.add_argument("--face")
    p_run.add_argument("--temp", type=float)
    p_run.add_argument("--noise", type=float)
    p_run.add_argument("--feed-gap", type=float)
    p_run.add_argument("--sleep-gap", type=float)
    p_run.add_argument("--context-field", action="append", metavar="K=V")
    p_run.add_argument("--sensor", action="append", metavar="K=V")

    p_feedback = sub.add_parser("feedback", help="record parent feedback")
    p_feedback.add_argument("--event", type=int, required=True)
    p_feedback.add_argument("--cause")
    p_feedback.add_argument("--correct", action="store_true")
    p_feedback.add_argument("--wrong", action="store_true")
    p_feedback.add_argument("--notes", default="")

    p_report = sub.add_parser("report", help="print pilot metrics")
    p_report.add_argument("--family")
    p_report.add_argument("--json", action="store_true")

    p_export = sub.add_parser("export", help="export anonymized events")
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--family")
    p_export.add_argument("--include-audio", action="store_true")
    p_export.add_argument("--include-notes", action="store_true")

    p_train = sub.add_parser(
        "export-training",
        help="copy audio-retained, feedback-labeled events into class folders",
    )
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--family")

    p_withdraw = sub.add_parser("withdraw", help="withdraw consent")
    p_withdraw.add_argument("--family", required=True)
    p_withdraw.add_argument("--keep-audio", action="store_true")

    sub.add_parser("families", help="list enrolled families")
    args = parser.parse_args()
    store = PilotStore(args.root)

    if args.command == "init":
        family_id = store.add_family(
            args.age_days, args.consent_version, args.audio_retention,
            args.notes, args.family_id,
        )
        print(f"family_id={family_id}")
        print(f"audio_retention={bool(args.audio_retention)}")
        print(f"consent_version={args.consent_version}")
        return

    if args.command == "families":
        for family in store.list_families():
            status = "withdrawn" if family["withdrawn"] else "active"
            print(f"{family['family_id']:18s} age={family['age_days']:>4d}d "
                  f"events={family['events']:>3d} audio={family['audio_retention']} "
                  f"{status}")
        return

    if args.command == "report":
        report = store.report(args.family)
        if args.json:
            print(json.dumps(report, indent=2, default=_json_default))
        else:
            _print_report(report)
        return

    if args.command == "export":
        manifest = store.export(
            args.out, args.family, args.include_audio, args.include_notes
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "export-training":
        manifest = store.export_training_set(args.out, args.family)
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "withdraw":
        store.withdraw(args.family, delete_audio=not args.keep_audio)
        print(f"withdrawn {args.family}")
        return

    if args.command == "feedback":
        if not args.cause and not args.correct and not args.wrong:
            raise SystemExit("Pass --cause, --correct, or --wrong")
        correct = True if args.correct else (False if args.wrong else None)
        store.record_feedback(args.event, args.cause, correct, args.notes)
        print(f"feedback recorded for event {args.event}")
        return

    if args.command == "run":
        from config import load_config
        from pipeline import CryAgentPipeline
        from baby_profile import BabyProfile

        family = store.get_family(args.family)
        if not family:
            raise SystemExit(f"Unknown family: {args.family}")
        if family["withdrawn"]:
            raise SystemExit(f"Family {args.family} has withdrawn consent")

        scenario = args.scenario
        pipeline = CryAgentPipeline(load_config(), scenario)
        profile = BabyProfile(age_days=family["age_days"])
        if args.feed_gap is not None:
            profile.last_feed_time = time.time() - args.feed_gap * 3600
        if args.sleep_gap is not None:
            profile.last_sleep_end = time.time() - args.sleep_gap * 3600

        result = pipeline.run(
            audio_path=args.audio,
            postures=[p.strip() for p in args.postures.split(",") if p.strip()],
            face_state=args.face,
            room_temp=args.temp,
            ambient_noise_db=args.noise,
            profile=profile,
            context_values=_pairs(args.context_field),
            env_values=_pairs(args.sensor),
        )
        event_id = store.log_result(args.family, result, args.audio)
        print(f"event_id={event_id}")
        print(f"cause={result.cause} confidence={result.confidence:.0%} "
              f"model={result.model} latency={result.latency_ms:.0f}ms")
        if args.audio and not family["audio_retention"]:
            print("audio not retained (family did not opt in)")
        if args.audio and family["audio_retention"]:
            print("audio retained locally for this family")


if __name__ == "__main__":
    main()
