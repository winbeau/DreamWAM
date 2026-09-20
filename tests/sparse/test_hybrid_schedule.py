"""Planning checks run with the standard library alone, including import isolation."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dreamwam.sparse.hybrid import HybridConfig, StepContext, compile_plan
from dreamwam.sparse.hybrid.schedule import Schedule
from dreamwam.sparse.hybrid.search import generate_candidates, generate_sweep, parse_indices, read_candidates


def options(n=10):
    return dict(schedule=dict(kind="periodic", num_steps=n, refresh_every=5),
                recompute=dict(keep_ratio=0.1), read=dict(mode="compact", keep_ratio=0.25))


class ScheduleTests(unittest.TestCase):
    def test_budget_selector_grid_preserves_schedule_and_canonical_identity(self):
        config = HybridConfig.from_mapping(options())
        rows = generate_sweep(config, (0,), (2, 5, 8), (1,), read_ratios=(0.25, 0.5),
                              recompute_ratios=(0.05, 0.1), selections=("uniform", "action_drift"))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["candidate_id"] for row in rows}), 24)
        for row in rows:
            effective = HybridConfig.from_mapping(row["options"])
            self.assertEqual(effective.policy_hash, row["candidate_id"])
            self.assertEqual(effective.schedule.operations.count("sparse"), 1)
        self.assertEqual(generate_sweep(config, (0,), (5,), (1,)),
                         list(generate_candidates(config, (0,), (5,), (1,))))

    def test_grid_rejects_bad_axes_and_bound_before_writing_any_manifest(self):
        config = HybridConfig.from_mapping(options())
        for kwargs in (dict(read_ratios=()), dict(read_ratios=(0.25, 0.25)),
                       dict(read_ratios=(0.25, float("nan"))), dict(read_ratios=(0.05,)),
                       dict(selections=("unknown",)), dict(recompute_ratios=(True,)),
                       dict(max_candidates=0), dict(max_candidates=1, read_ratios=(0.25, 0.5))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_sweep(config, (0,), (5,), (1,), **kwargs)
        with self.assertRaisesRegex(ValueError, "empty schedule"):
            generate_sweep(config, (0,), (5,), ())

    def test_grid_cli_is_finite_exclusive_and_roundtrips(self):
        script = Path(__file__).resolve().parents[2] / "scripts/sparse/generate_hybrid_schedules.py"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.jsonl"
            command = [sys.executable, str(script), "--candidate-sparse-steps", "5",
                       "--refresh-counts", "1", "--read-ratios", "0.125,0.25,0.5,0.75",
                       "--backend", "cuda_graph", "--out", str(path)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(len(read_candidates(path)), 4)
            original = path.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(path.read_bytes(), original)
            invalid = Path(tmp) / "invalid.jsonl"
            for extra in (["--read-ratio", "0.5"], ["--max-candidates", "2"]):
                self.assertNotEqual(subprocess.run(command[:-1] + [str(invalid)] + extra,
                                                    capture_output=True).returncode, 0)
                self.assertFalse(invalid.exists())

    def test_exact_search_spaces_and_nonperiodic_last_step(self):
        config = HybridConfig.from_mapping(options())
        small = list(generate_candidates(config, (0,), tuple(range(1, 10)), (0, 1, 2)))
        full = list(generate_candidates(config, (0,), tuple(range(1, 10)), tuple(range(10))))
        self.assertEqual(len(small), 46)
        self.assertEqual(len(full), 512)
        self.assertEqual(len({c["candidate_id"] for c in full}), 512)
        self.assertTrue(any(c["sparse_steps"] == [1, 9] for c in small))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in small))
            self.assertEqual(len(read_candidates(path)), 46)
            path.write_text(json.dumps(small[0]) + "\n" + json.dumps(small[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_candidates(path)

    def test_canonical_identity_and_actual_step_validation(self):
        periodic = HybridConfig.from_mapping(options())
        explicit = replace(periodic, schedule=Schedule(10, periodic.schedule.operations))
        self.assertEqual(periodic.policy_hash, explicit.policy_hash)
        plan = compile_plan(periodic, 10)
        self.assertEqual(plan.decision(StepContext(0, 10)).requested_q_ratio, 1)
        self.assertEqual(plan.decision(StepContext(5, 10)).operation, "sparse")
        self.assertEqual(plan.decision(StepContext(9, 10)).requested_q_ratio, 0)
        for n in (4, True):
            with self.assertRaises(ValueError):
                compile_plan(periodic, n)
        with self.assertRaises(ValueError):
            plan.decision(StepContext(10, 10))

    def test_strict_configuration_and_boundaries(self):
        for patch in (dict(typo=True), dict(schema_version=True), dict(read=dict(mode="full", keep_ratio=0.1)),
                      dict(recompute=dict(keep_ratio=True)), dict(recompute=dict(keep_ratio=0)),
                      dict(recompute=dict(keep_ratio=float("nan"))),
                      dict(recompute=dict(keep_ratio=0.5)), dict(execution=dict(backend="silent_fallback"))):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                HybridConfig.from_mapping({**options(), **patch})
        for operations in (["reuse"] * 10, ["dense"], ["dense"] + ["typo"] * 9):
            with self.assertRaises(ValueError):
                HybridConfig.from_mapping({**options(), "schedule": dict(kind="explicit", num_steps=10,
                                                                        operations=operations)})
        config = HybridConfig.from_mapping(options())
        self.assertEqual(config.budgets(294, 98), (30, 74))
        with self.assertRaises(ValueError):
            replace(config, read_ratio=0.1).budgets(12, 4)
        for indices in ("1,1", "1:3,2", "3:1", "-1"):
            with self.assertRaises(ValueError):
                parse_indices(indices)
        with self.assertRaises(ValueError):
            list(generate_candidates(config, (0,), (0, 1), (1,)))
        self.assertEqual(parse_indices(""), ())
        dense = list(generate_candidates(config, tuple(range(10)), (), (0,)))
        self.assertEqual(len(dense), 1)
        self.assertEqual(dense[0]["options"]["schedule"]["operations"], ["dense"] * 10)

    def test_profile_requires_hash_budget_and_layout_agreement(self):
        config = HybridConfig.from_mapping(options())
        compatibility = dict(video_length=294, tokens_per_frame=98, scheduler_hash="a" * 64)
        doc = dict(schema_version=1, num_steps=10, operations=list(config.schedule.operations),
                   policy_hash=config.policy_hash, compatibility=compatibility)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            data = json.dumps(doc).encode()
            path.write_bytes(data)
            payload = {**config.describe(), "schedule": dict(kind="profile", num_steps=10,
                        path=str(path), sha256=hashlib.sha256(data).hexdigest())}
            loaded = HybridConfig.from_mapping(payload)
            self.assertEqual(compile_plan(loaded, 10, compatibility=compatibility).plan_hash, config.policy_hash)
            with self.assertRaisesRegex(ValueError, "layout"):
                compile_plan(loaded, 10, compatibility={**compatibility, "video_length": 98})
            with self.assertRaisesRegex(ValueError, "policy"):
                compile_plan(replace(loaded, read_ratio=0.5), 10)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                HybridConfig.from_mapping(payload)

    def test_imports_do_not_load_torch_or_model(self):
        subprocess.run([sys.executable, "-c",
            "import sys; from dreamwam.sparse.hybrid import HybridConfig; "
            "assert 'torch' not in sys.modules; assert 'dreamwam.model' not in sys.modules"],
            check=True)


if __name__ == "__main__":
    unittest.main()
