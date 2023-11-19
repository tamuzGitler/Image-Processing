import unittest
import numpy as np
import sol2 as sol2
import scipy.io.wavfile

class MyTestCase(unittest.TestCase):
    def test_change_samples(self):
        wav_data_orig = scipy.io.wavfile.read(r"aria_4kHz.wav")

        # ratio = 0.25
        # print("starting test with ratio parameter:", ratio)
        # returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        # wav_data_025 = scipy.io.wavfile.read(r"change_samples.wav")
        #
        # self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        # self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_025[1]))
        # print("finished test with ratio parameter:", ratio)

        ratio = 0.8
        print("starting test with ratio parameter:", ratio)
        returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        wav_data_08 = scipy.io.wavfile.read(r"change_samples.wav")
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_08[1]))
        print("finished test with ratio parameter:", ratio)

        ratio = 1
        print("starting test with ratio parameter:", ratio)
        returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        wav_data_1 = scipy.io.wavfile.read(r"change_samples.wav")
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_1[1]))
        print("finished test with ratio parameter:", ratio)

        ratio = 2
        print("starting test with ratio parameter:", ratio)
        returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        wav_data_2 = scipy.io.wavfile.read(r"change_samples.wav")
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_2[1]))
        print("finished test with ratio parameter:", ratio)

        ratio = 3
        print("starting test with ratio parameter:", ratio)
        returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        wav_data_3 = scipy.io.wavfile.read(r"change_samples.wav")
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_3[1]))
        print("finished test with ratio parameter:", ratio)

        ratio = 4
        print("starting test with ratio parameter:", ratio)
        returned_data = sol2.change_samples(r"aria_4kHz.wav", ratio)
        wav_data_4 = scipy.io.wavfile.read(r"change_samples.wav")
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(returned_data))
        self.assertEqual(int(len(wav_data_orig[1]) / ratio), len(wav_data_4[1]))
        print("finished test with ratio parameter:", ratio)


if __name__ == '__main__':
    unittest.main()
