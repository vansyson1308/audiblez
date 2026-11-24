"""Choice of the recommended configuration by audiblez-bench."""
import unittest

from audiblez.bench import recommend


def r(backend, dtype, bs, x, passed, cer=0.02, error=None, problems=()):
    return dict(backend=backend, dtype=dtype, batch_size=bs, x_realtime=x, passed=passed, total=8, mean_cer=cer,
                error=error, problems=list(problems))


class RecommendTest(unittest.TestCase):
    def test_fastest_with_equal_quality(self):
        results = [r('standard', 'float32', 1, 0.45, 8), r('standard', 'float32', 8, 1.9, 8),
                   r('fast', 'float32', 1, 0.9, 8)]
        self.assertEqual(recommend(results)['batch_size'], 8)

    def test_quality_beats_speed(self):
        results = [r('standard', 'float32', 1, 0.45, 8), r('fast', 'float16', 1, 2.0, 4),  # many failed chunks
                   r('fast', 'float32', 1, 0.9, 8, cer=0.02), r('standard', 'float16', 8, 2.5, 8, cer=0.12)]
        best = recommend(results)
        self.assertEqual((best['backend'], best['dtype']), ('fast', 'float32'))

    def test_errors_and_nan_are_excluded(self):
        results = [r('fast', 'float16', 1, 3.0, 8, problems=['NaN/inf in the audio']),
                   r('standard', 'float16', 8, None, None, error='RuntimeError: probability tensor contains inf'),
                   r('standard', 'float32', 4, 1.2, 7)]
        self.assertEqual(recommend(results)['dtype'], 'float32')
        self.assertIsNone(recommend([r('fast', 'float16', 1, None, None, error='OOM')]))


if __name__ == '__main__':
    unittest.main()
