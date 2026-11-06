"""Offline checks of the request-concurrency and timing measurement contract."""
import asyncio
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from evals.controlled import percentile, reject_suspension, sample_batch, workload


class FakeSampler:
    def __init__(self):
        self.active = 0
        self.maximum = 0
        self.finished = 0

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.002)
        self.active -= 1
        self.finished += 1
        return SimpleNamespace(
            sequences=[SimpleNamespace(sequence_id=f'{self.finished}-{i}', tokens=[3, 4, 5],
                                       stop_reason='stop') for i in range(num_samples)],
            prompt_cache_hit_tokens=0,
        )


class ControlledMeasurementTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_actual_sequences_and_waits_for_every_request(self):
        sampler = FakeSampler()
        prompts = [dict(key=str(i), input_ids=[1, 2]) for i in range(7)]
        summary, records = await sample_batch(sampler, prompts, None, 4, 2,
            dict(backend='offline-test', model='fake', phase='test'))
        self.assertEqual(sampler.maximum, 2)
        self.assertEqual(sampler.finished, 7)
        self.assertEqual(summary['sample_tokens'], 7*4*3)
        self.assertEqual(summary['sequence_count'], 28)
        self.assertEqual(summary['max_offered_sequence_concurrency'], 8)
        self.assertEqual(len(records), 7)
        self.assertTrue(all(len(r['sequences']) == 4 for r in records))
        self.assertEqual([r['index'] for r in records], list(range(7)))
        self.assertGreater(summary['wall_s'], 0)
        self.assertLessEqual(summary['last_request_completion_s'], summary['wall_s'])
        self.assertTrue(all(r['complete_offset_s'] <= summary['wall_s'] for r in records))
        self.assertAlmostEqual(summary['output_tokens_per_second'], 84/summary['wall_s'])
        self.assertIsNone(summary['ttft_s'])
        self.assertTrue(all(r['first_token_at'] is None for r in records))

    async def test_one_request_at_a_time(self):
        sampler = FakeSampler()
        prompts = [dict(key=str(i), input_ids=[1]) for i in range(4)]
        summary, _ = await sample_batch(sampler, prompts, None, 1, 1,
            dict(backend='offline-test', model='fake', phase='test'))
        self.assertEqual(sampler.maximum, 1)
        self.assertEqual(summary['sample_tokens'], 12)
        self.assertLessEqual(summary['request_latency_p50_s'], summary['request_latency_p95_s'])

    def test_rejects_host_sleep_instead_of_reporting_inflated_throughput(self):
        with patch('evals.controlled.elapsed_clock', return_value=100), patch('evals.controlled.time.monotonic', return_value=20):
            with self.assertRaisesRegex(RuntimeError, 'Host suspended'):
                reject_suspension(10, 10)

    def test_autoscaling_defaults_and_reference_workload(self):
        from spindle_configs.qwen35_4b import config as small
        from spindle_configs.qwen35_9b import config as medium
        from spindle_configs.nemotron35_lightning import config as large
        for cfg,maximum,target in [(small,8,32),(medium,8,16),(large,4,64)]:
            self.assertEqual((cfg.inference_min_replicas,cfg.inference_max_replicas),(0,maximum))
            self.assertEqual(cfg.inference_target_concurrency,target)
            self.assertEqual((cfg.pool_idle_timeout_s,cfg.session_idle_timeout_s),(300,300))
        data=workload()
        self.assertEqual((len(data['ifeval']['rows']),len(data['math500']['rows'])),(128,64))
        self.assertEqual((data['ifeval']['num_samples'],data['math500']['num_samples']),(1,4))

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], .5), 3)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], .95), 4.8)


if __name__ == '__main__':
    unittest.main()
