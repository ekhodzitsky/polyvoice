#!/usr/bin/env python3
"""Fail-closed checks for held-out corpus evaluation."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).with_name('notsofar-eval.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_transient_download_retries_but_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.write_bytes(b'test')
            manifest = {'repository': 'https://example.test', 'revision': 'frozen', 'files': [],
                        'license': {'path': 'LICENSE.txt', 'size': 4, 'sha256': evaluation.sha(source)}}
            with patch.object(evaluation.urllib.request, 'urlopen', side_effect=[
                    evaluation.http.client.RemoteDisconnected(), io.BytesIO(b'test')]) as fetch, \
                 patch.object(evaluation.time, 'sleep'):
                evaluation.download(manifest, root / 'download')
                self.assertEqual(fetch.call_count, 2)
            with patch.object(evaluation.urllib.request, 'urlopen', return_value=io.BytesIO(b'evil')) as fetch:
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    evaluation.download(manifest, root / 'corrupt')
                self.assertEqual(fetch.call_count, 1)
            self.assertFalse((root / 'corrupt/LICENSE.txt').exists())
            self.assertFalse((root / 'corrupt/LICENSE.txt.tmp').exists())

    def setUp(self):
        self.manifest = {'files': [{'id': 'meeting'}], 'regression_tolerance_pp': 2.0}
        self.report = {'manifest_sha256': 'manifest', 'scorer_sha256': 'scorer', 'protocol': {'collar_seconds': 0},
                       'model_hashes': {'segmenter': 'model'}, 'per_file': [{'name': 'meeting'}],
                       'der_micro': 30.0, 'der_macro': 35.0}
        self.baseline = copy.deepcopy(self.report)

    def test_threshold_boundaries_and_both_aggregates(self):
        for key in ['der_micro', 'der_macro']:
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                report[key] += 2.0
                evaluation.check_report(report, self.baseline, self.manifest)
                report[key] += 0.0001
                with self.assertRaisesRegex(ValueError, 'exceeds'):
                    evaluation.check_report(report, self.baseline, self.manifest)

    def test_incomplete_duplicate_and_wrong_file_sets_fail(self):
        for names in [[], ['other'], ['meeting', 'meeting']]:
            report = dict(self.report, per_file=[{'name': n} for n in names])
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'coverage'):
                evaluation.check_report(report, self.baseline, self.manifest)

    def test_nonfinite_scores_and_changed_protocol_fail(self):
        for key, value in [('der_micro', float('nan')), ('der_macro', float('inf')),
                           ('model_hashes', {}), ('manifest_sha256', 'other'), ('scorer_sha256', 'other'),
                           ('protocol', {'collar_seconds': 0.25})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluation.check_report(dict(self.report, **{key: value}), self.baseline, self.manifest)

    def test_missing_corrupted_and_extra_inputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ['audio', 'gt']:
                (root / folder).mkdir()
            paths = [root / 'LICENSE.txt', root / 'audio/meeting.wav', root / 'gt/meeting.json']
            for path in paths:
                path.write_bytes(b'test')
            entry = {'size': 4, 'sha256': evaluation.sha(paths[0])}
            manifest = {'license': entry, 'files': [{'id': 'meeting', 'audio': entry, 'reference': entry}]}
            evaluation.validate_dataset(manifest, root)
            paths[1].unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.validate_dataset(manifest, root)
            paths[1].write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                evaluation.validate_dataset(manifest, root)
            paths[1].write_bytes(b'test')
            (root / 'audio/extra.wav').write_bytes(b'test')
            with self.assertRaisesRegex(ValueError, 'coverage'):
                evaluation.validate_dataset(manifest, root)

    def test_empty_hypothesis_scores_all_reference_as_missed(self):
        hyp = evaluation.hypothesis_turns({'turns': [], 'num_speakers': 0}, 10)
        score = evaluation.der.score_file([(0, 10, 'a')], hyp, collar=0)
        self.assertEqual(score.der, 100)
        self.assertEqual(score.miss, score.scored_ref)

    def test_speaker_ids_need_not_be_contiguous_but_count_must_match(self):
        result = {'num_speakers': 2, 'turns': [
            {'speaker': 4, 'time': {'start': 0, 'end': 1}},
            {'speaker': 7, 'time': {'start': 1, 'end': 2}}]}
        self.assertEqual(evaluation.hypothesis_turns(result, 2), [(0, 1, '4'), (1, 2, '7')])
        result['num_speakers'] = 3
        with self.assertRaisesRegex(ValueError, 'speaker count'):
            evaluation.hypothesis_turns(result, 2)

    def test_reference_preserves_overlap_and_rejects_bad_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gt.json'
            data = [{'speaker_id': 'a', 'start_time': 0, 'end_time': 2},
                    {'speaker_id': 'b', 'start_time': 1, 'end_time': 3}]
            path.write_text(json.dumps(data))
            self.assertEqual(evaluation.reference_turns(path, 3), [(0, 2, 'a'), (1, 3, 'b')])
            data[0]['end_time'] = 30
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                evaluation.reference_turns(path, 3)


if __name__ == '__main__':
    unittest.main()
