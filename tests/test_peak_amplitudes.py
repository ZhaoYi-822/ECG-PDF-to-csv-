import unittest
import numpy as np
from PIL import Image, ImageDraw
from ecg_tool.extractor import _waveform_likelihood, _waveform_color_mask, _trace_waveform_subpixel


def raster_trace(points, width=4):
    scale=4
    image=Image.new('RGB',(200*scale,210*scale),'white')
    drawing=ImageDraw.Draw(image)
    drawing.line([(int(x*scale),int(y*scale)) for x,y in points], fill=(255,128,70), width=width*scale, joint='curve')
    return np.asarray(image.resize((200,210),Image.Resampling.LANCZOS))


class PeakAmplitudeTests(unittest.TestCase):
    def test_narrow_positive_and_negative_peaks_keep_known_amplitudes(self):
        for half_width in (2,3,5):
            for stroke in (3,5):
                with self.subTest(half_width=half_width,stroke=stroke):
                    image=raster_trace([(0,120),(60-half_width,120),(60,40),(60+half_width,120),(130-half_width,120),(130,170),(130+half_width,120),(199,120)],stroke)
                    x,y,valid,_=_trace_waveform_subpixel(_waveform_likelihood(image),_waveform_color_mask(image))
                    # Ground truth: +0.8 mV and -0.5 mV at 100 px/mV.
                    recovered_r=(120-min(y[55:66]))/100
                    recovered_s=(120-max(y[125:136]))/100
                    self.assertAlmostEqual(recovered_r,.8,delta=.035)
                    self.assertAlmostEqual(recovered_s,-.5,delta=.035)
                    # No blanket gain change or baseline displacement is allowed.
                    self.assertLess(abs(float(np.median(y[10:40]))-120),1.0)
                    self.assertTrue(valid[60])

    def test_peak_survives_500_hz_resampling(self):
        image=raster_trace([(0,120),(58,120),(60,40),(62,120),(199,120)])
        x,y,_,_=_trace_waveform_subpixel(_waveform_likelihood(image),_waveform_color_mask(image))
        # 300 native columns/s -> output at 500 Hz, matching the report density.
        sample_x=np.arange(334)*300/500
        yy=np.interp(sample_x,x,y)
        self.assertAlmostEqual((120-float(yy.min()))/100,.8,delta=.035)
