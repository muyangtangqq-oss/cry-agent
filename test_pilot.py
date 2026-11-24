"""End-to-end test of the local pilot harness with 5 simulated families."""
import csv
import os
import tempfile
from types import SimpleNamespace

from pilot import PilotStore


def fake_result(cause, confidence=0.85, model="fusion", latency=180.0,
                ood=False):
    return SimpleNamespace(
        cause=cause,
        confidence=confidence,
        model=model,
        fused_cause=cause,
        fused_confidence=confidence,
        audio_cause=cause,
        audio_confidence=confidence,
        vision_cause="unknown",
        vision_confidence=0.0,
        escalated=False,
        critical_override=False,
        audio_ood=ood,
        session_ood=ood,
        latency_ms=latency,
        trusted=not ood,
        scenario="baby_cry",
        per_signal={"audio": {"value": cause}},
    )


def run():
    with tempfile.TemporaryDirectory() as tmp:
        store = PilotStore(tmp)
        families = []
        for i in range(5):
            families.append(store.add_family(
                age_days=30 + i * 10,
                audio_retention=(i < 2),
                notes=f"volunteer {i}",
            ))
        assert len(store.list_families()) == 5

        causes = ["hungry", "sleepy", "discomfort", "burp", "pain"]
        event_ids = []
        for i in range(20):
            family = families[i % len(families)]
            audio_path = None
            if i < 2:
                audio_path = os.path.join(tmp, f"clip_{i}.wav")
                with open(audio_path, "wb") as f:
                    f.write(b"RIFFfake")
            event_id = store.log_result(
                family,
                fake_result(causes[i % len(causes)], latency=100 + i * 10),
                audio_path=audio_path,
            )
            event_ids.append(event_id)

        assert len(event_ids) == 20
        for i, event_id in enumerate(event_ids):
            if i % 4 == 0:
                continue
            store.record_feedback(
                event_id,
                actual_cause=causes[i % len(causes)],
            )
        store.record_feedback(event_ids[0], actual_cause="sleepy")

        report = store.report()
        assert report["events"] == 20
        assert report["families"] == 5
        assert report["feedback_events"] == 16
        assert report["audio_retained"] == 2
        assert report["ood_rate"] == 0
        assert report["untrusted_rate"] == 0
        assert report["latency_ms"]["p50"] is not None
        assert report["accuracy_on_feedback"] < 1.0
        assert len(report["by_family"]) == 5

        out = os.path.join(tmp, "export")
        store.export(out, include_audio=True)
        with open(os.path.join(out, "events.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20
        assert all(not row["family_hash"].startswith("fam_") for row in rows)
        assert len(os.listdir(os.path.join(out, "audio"))) == 2
        assert os.path.exists(os.path.join(out, "manifest.json"))

        train_out = os.path.join(tmp, "training_set")
        train_manifest = store.export_training_set(train_out)
        assert train_manifest["clips"] == 2
        assert sum(train_manifest["per_class"].values()) == 2
        assert os.path.exists(os.path.join(train_out, "manifest.json"))

        store.withdraw(families[0])
        assert store.get_family(families[0])["withdrawn"] == 1
        assert not os.path.isdir(os.path.join(tmp, "audio", families[0]))

        try:
            store.log_result(families[0], fake_result("hungry"))
        except ValueError:
            pass
        else:
            raise AssertionError("withdrawn family accepted a new event")

        print("5 simulated families")
        print(f"events=20 feedback=16 accuracy="
              f"{report['accuracy_on_feedback']:.0%}")
        print(f"export rows={len(rows)} audio=2 training_clips=2")
        print("withdrawal deleted audio and blocked new events")

        shifted = []
        for _ in range(8):
            shifted.append(store.log_result(
                families[2], fake_result("hungry", ood=True)
            ))
        assert store.session_ood(families[2])
        store.mark_untrusted(shifted[-1])
        assert store.get_event(shifted[-1])["trusted"] == 0
        shift_report = store.report()
        assert families[2] in shift_report["domain_shift_families"]
        print("domain shift detected after 8 OOD events")


if __name__ == "__main__":
    run()
    print("Pilot harness test passed.")
