"""Offline regression checks. Never imports XRT or submits to the NPU.

Run with scripts/iron_python.ps1 experiments/012_schedule_check.py.
Uses only packed-weight metadata; no weight/position packing or device buffers.
"""
import json
import sys
import unittest
from unittest.mock import patch

# Fail closed if a code change accidentally introduces a hardware path.
sys.modules["pyxrt"] = None
sys.modules["npu_direct"] = None

import pk_engine as pk
import pk_model as pm


def metadata_plan(nblk):
    lay = pk.Layout(nblk)
    table = json.loads((pm.PKDIR / "weights.json").read_text())
    size = pk.DK * 2 * lay.TPAD
    pos_offs = [[(i * pk.HEADS + h) * size for h in range(pk.HEADS)] for i in range(24)]
    phases = pm.build_phases(lay, table, 24, pos_offs)
    lens = dict(ctrl=pk.assign_headers(phases), weights=table["size"],
                pos=24 * pk.HEADS * size, io=lay.io_len)
    return lay, phases, lens


class ScheduleTests(unittest.TestCase):
    def test_full_encoders(self):
        for nblk in (1, 2, 3):
            with self.subTest(nblk=nblk):
                lay, phases, lens = metadata_plan(nblk)
                self.assertEqual(len(phases), 529)
                self.assertEqual(pk.check(phases, lay, lens), [])
                per = pm.LAYERS_PER_RUN[nblk]
                cuts = [0] + [1 + 22 * k for k in range(per, 24, per)] + [len(phases)]
                for a, b in zip(cuts, cuts[1:]):
                    self.assertEqual(pk.check(phases[a:b], lay, lens), [])

    def test_attention_larger_groups_rejected(self):
        for passes in (2, 3, 4):
            with self.subTest(passes=passes), patch.object(pk, "ATT_GROUP", passes):
                lay, phases, lens = metadata_plan(2)
                self.assertTrue(any("queue depth 4" in p for p in pk.check(phases, lay, lens)))

    def test_old_attention_group_rejected(self):
        lay, phases, lens = metadata_plan(1)
        att = next(p for p in phases if isinstance(p, pk.Att))
        original = att.groups

        def old_groups(lay, c):
            prologue, p0, p1, p2, p3 = original(lay, c)
            # Recreate the old first two-pass group: 3 + 3 + 3 W tasks,
            # still only 10 BDs with the combined drain, so MAX_BDS passed.
            d = list(p0[2][0])
            d[2] = [4, *d[2][1:]]
            return [(prologue[0] + p0[0] + p1[0], [], [tuple(d)]), p2, p3]

        with patch.object(att, "groups", old_groups):
            errors = pk.check([att], lay, lens)
        self.assertTrue(any("9 unretired task starts" in p for p in errors))
        self.assertFalse(any("descriptors >" in p for p in errors))
        self.assertFalse(any("core expects" in p for p in errors))

    def test_unsafe_order_rejected_by_check(self):
        lay, phases, lens = metadata_plan(1)
        original = pk.group_tasks

        def fills_first(columns):
            return sorted(original(columns), key=lambda t: t[0] == "C")

        with patch.object(pk, "group_tasks", fills_first):
            errors = pk.check(phases[:2], lay, lens)
        self.assertTrue(any("fill issued before" in p for p in errors))

    def test_queue_counts_starts_not_repeats(self):
        task = ("io", 0, [128, 1, 1, pk.WOBJ], [0, 0, 0, 1])
        starts = [("W", 0, task, True)] * 4
        self.assertEqual(pk.check_task_order(starts), [])
        self.assertTrue(any("queue depth" in p for p in pk.check_task_order(starts * 2)))
        # Different columns and directions/channels have independent queues.
        self.assertEqual(pk.check_task_order([
            (kind, c, task, True) for kind in ("C", "W", "X")
            for c in range(4) for _ in range(4)]), [])

    def test_missing_prologue_wait_rejected(self):
        lay, phases, lens = metadata_plan(1)
        att = next(p for p in phases if isinstance(p, pk.Att))
        original = pk.group_tasks
        with patch.object(pk, "group_tasks", lambda cols: [
                (k, c, t, wait if k == "C" else False)
                for k, c, t, wait in original(cols)]):
            errors = pk.check([att], lay, lens)
        self.assertTrue(any("input-only group needs" in p for p in errors))

    def test_group_boundary_requires_matching_inputs(self):
        lay, phases, lens = metadata_plan(1)
        att = next(p for p in phases if isinstance(p, pk.Att))
        original = att.groups

        def early_drain(lay, c):
            groups = original(lay, c)
            a, b = groups[1], groups[2]
            # Total counts preserved, but the first pass waits before its KV
            # transfer has been issued. Both groups remain within four pushes.
            groups[1] = (a[0][:-1], a[1], a[2])
            groups[2] = ([a[0][-1]] + b[0], b[1], b[2])
            return groups

        with patch.object(att, "groups", early_drain):
            errors = pk.check([att], lay, lens)
        self.assertTrue(any("group completion has" in p for p in errors))
        self.assertFalse(any("core expects" in p for p in errors))

    def test_reject_before_hardware_import(self):
        lay, phases, lens = metadata_plan(2)
        # Supply a metadata-only plan; no weight/position arrays needed before
        # the checker rejects the schedule. npu_direct is blocked above.
        with patch.object(pk, "ATT_GROUP", 2), patch.object(pm, "plan", return_value=(
                lay, phases, [], {"size": lens["weights"]})):
            with self.assertRaisesRegex(ValueError, "no device opened"):
                pm.PkEncoder(nblk=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
