import unittest
from unittest.mock import patch
import threading
import time
import numpy as np

from stitch_blade_refined import bounded_parallel_map
from sam2_alignment import (mask_geometry, sample_curve, contour_constraint, transform,
                            project_corrected_grid,
                            robust_residual, smooth_error, refine, evaluate, MaskBlender, solve)


def curves(offset=0., width=2.):
    v = np.linspace(0, 10, 500)
    return [np.column_stack((np.full_like(v, offset-width/2), v)),
            np.column_stack((np.full_like(v, offset+width/2), v))]


class MaskGeometryTests(unittest.TestCase):
    def test_off_center_wide_blade_and_holes_keep_actual_outer_boundary(self):
        mask = np.zeros((240, 320), np.uint8)
        mask[:, 20:240] = 255
        mask[40:60, 80:100] = 0
        g = mask_geometry(mask)
        np.testing.assert_array_equal(g['left'][:, 0], 20)
        np.testing.assert_array_equal(g['right'][:, 0], 239)
        self.assertGreater(g['left'][:, 1].min(), 2)
        self.assertLess(g['left'][:, 1].max(), 237)
        self.assertEqual(g['outline'][:, 1].max(), 239)

    def test_missing_boundary_segment_is_not_fabricated(self):
        y = np.r_[np.linspace(0, 1, 50), np.linspace(3, 4, 50)]
        curve = np.column_stack((y*.1, y))
        _, valid = sample_curve(curve, np.array([.5, 2., 3.5]))
        np.testing.assert_array_equal(valid, [True, False, True])


class RegistrationTests(unittest.TestCase):
    def test_transform_inverse_and_pivot(self):
        xy = np.array([[-2., 4.], [4., 12.], [.5, -8.]])
        q = np.array([.4, -.5, .007, .02]); pivot = np.array([.2, 3.])
        np.testing.assert_allclose(transform(transform(xy, q, pivot), q, pivot, inverse=True), xy)
        np.testing.assert_allclose(transform(pivot, q, pivot), pivot+q[:2])

    def test_scalar_grid_projection_matches_reference_projection(self):
        from blade_refinement import Surface
        surface=Surface(np.array([1.,2.,3.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]),
                        np.array([0.,0.,1.]),np.array([.001,.002,12.]))
        q=np.array([.3,-.2,.004,.015]);pivot=np.array([.1,4.])
        C=np.array([.2,-1.,.5]);R=np.eye(3);K=np.array([[1000.,2.,640.],[0.,990.,480.],[0.,0.,1.]])
        u=np.linspace(-2,2,11)[None,:];v=np.linspace(-8,8,9)[:,None]
        U,V=np.broadcast_arrays(u,v)
        source=transform(np.stack((U,V),axis=-1),q,pivot,inverse=True)
        expected,front=surface.project(source,C,R,K)
        x,y,actual_front=project_corrected_grid(u,v,q,pivot,surface,C,R,K)
        np.testing.assert_allclose(x,expected[...,0],rtol=1e-13,atol=1e-12)
        np.testing.assert_allclose(y,expected[...,1],rtol=1e-13,atol=1e-12)
        np.testing.assert_array_equal(actual_front,front)

    def test_signed_edge_errors_do_not_cancel(self):
        e = np.array([-.2, .2])
        self.assertGreater(np.sum(robust_residual(e, .025)**2), 0)
        self.assertAlmostEqual(.5*np.sum(robust_residual(e, .025)**2), smooth_error(e)/.025)

    def test_parallel_edges_leave_longitudinal_motion_unobserved(self):
        link = contour_constraint(0, 1, curves(), curves(.3))
        q, records, info = refine(3, [link], np.zeros((3, 2)))
        self.assertAlmostEqual(q[0, 0]-q[1, 0], .3, delta=.01)
        np.testing.assert_array_equal(q[:, 1], 0)
        np.testing.assert_array_equal(q[2], 0)
        self.assertLess(records[0]['after_holdout_median_m'], .01)

    def test_affine_model_selected_only_when_it_improves_contours(self):
        pivot = np.array([0., 5.]); ci = curves()
        wanted = np.array([.2, 0., .006, .018])
        cj = [transform(c, wanted, pivot) for c in ci]
        link = contour_constraint(0, 1, ci, cj)
        _, baseline, _ = refine(2, [link], np.array([pivot, pivot]), model='translation')
        q, records, info = refine(2, [link], np.array([pivot, pivot]))
        self.assertEqual(info['groups'][0]['model'], 'lateral-affine')
        self.assertLess(records[0]['after_robust_m'], .95*baseline[0]['after_robust_m'])
        self.assertGreater(q[0, 2], 0.)
        self.assertLess(q[0, 2], .01)

    def test_failure_does_not_remove_independent_group(self):
        links = [contour_constraint(0, 1, curves(), curves(.2)),
                 contour_constraint(2, 3, curves(), curves(.3))]
        def failed(count, links, pivots, max_shift, **kwargs):
            if links[0]['i'] == 0:return None, 'synthetic failure'
            return solve(count, links, pivots, max_shift, **kwargs)
        with patch('sam2_alignment.solve', side_effect=failed):
            q, records, info = refine(4, links, np.zeros((4, 2)))
        np.testing.assert_array_equal(q[:2], 0)
        self.assertFalse(records[0]['accepted'])
        self.assertAlmostEqual(q[2, 0]-q[3, 0], .3, delta=.01)

    def test_lost_overlap_fails_validation(self):
        link = contour_constraint(0, 1, curves(), curves(.2))
        q = np.zeros((2, 4)); q[0, 1] = 20
        self.assertIn('validation_issue', evaluate(link, q, np.zeros((2, 2))))

    def test_shortened_overlap_retains_physical_support(self):
        link = contour_constraint(0, 1, curves(), curves(.2))
        q = np.zeros((2, 4)); q[0, 1] = 2.
        record = evaluate(link, q, np.zeros((2, 2)))
        self.assertNotIn('validation_issue', record)
        self.assertGreater(record['support_fraction'], .5)
        self.assertLess(record['support_fraction'], .9)


class MaskBlendTests(unittest.TestCase):
    def test_background_does_not_tint_foreground_or_expand_owner_mask(self):
        blend = MaskBlender((1, 2))
        alpha = np.array([[1., 0.]], np.float32)
        white = np.full((1, 2, 3), 200., np.float32)
        green = np.zeros((1, 2, 3), np.float32); green[..., 1] = 100
        blend.add(white*alpha[..., None], green*(1-alpha[..., None]), alpha, np.ones((1, 2), np.float32))
        other = 1-alpha
        blend.add(white*other[..., None], green*(1-other[..., None]), other, np.full((1, 2), .5, np.float32))
        output, valid = blend.finish()
        np.testing.assert_array_equal(output[0, 0], [200, 200, 200])
        np.testing.assert_array_equal(output[0, 1], [0, 100, 0])
        self.assertTrue(valid.all())

    def test_empty_tile_is_white_and_invalid(self):
        out, valid = MaskBlender((2, 3)).finish()
        self.assertTrue((out == 255).all())
        self.assertFalse(valid.any())


class ParallelRenderTests(unittest.TestCase):
    def test_bounded_parallel_map_preserves_order_and_runs_concurrently(self):
        lock = threading.Lock()
        active = peak = 0
        def work(value):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.02)
            with lock:
                active -= 1
            return value*value
        results = list(bounded_parallel_map(work, range(6), 2))
        self.assertEqual(results, [(i, i*i) for i in range(6)])
        self.assertEqual(peak, 2)


if __name__ == '__main__':
    unittest.main()
