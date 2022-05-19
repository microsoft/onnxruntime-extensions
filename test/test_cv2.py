import unittest
import numpy as np
from PIL import Image
from onnxruntime_extensions import get_test_data_file, OrtPyFunction

class TestOpenCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_image_reader(self):
        img_file = get_test_data_file('data', 'pineapple.jpg')
        rdr = OrtPyFunction.from_customop("ImageReader")
        nhwc_y = rdr([img_file])
        self.assertEqual(nhwc_y.shape[0], 1)
        self.assertEqual(nhwc_y.shape[3], 3)
        actual = nhwc_y.squeeze().transpose((2, 0, 1))
        actual = np.stack((actual[2], actual[1], actual[0]))
        actual = actual.transpose((1, 2, 0))

        # crossing check with pillow image API.
        pyimg = Image.open(img_file).convert('RGB')
        expected = np.asarray(pyimg)
        np.testing.assert_array_equal(actual, expected)

    def test_gaussian_blur(self):
        img_file = get_test_data_file('data', 'pineapple.jpg')
        img = Image.open(img_file).convert('RGB')
        img_arr = np.asarray(img, dtype=np.float32) / 255.
        img_arr = np.expand_dims(img_arr, 0)
        gb = OrtPyFunction.from_customop('GaussianBlur')
        gb_img = gb(img_arr, np.array([3, 3], dtype=np.int64), np.array([0.0, 0.0]))
        convimg = Image.fromarray((np.squeeze(gb_img, 0) * 255).astype(np.uint8), "RGB")
        # convimg.save('temp_pineapple.jpg')
        self.assertFalse(np.allclose(np.asarray(img), np.asarray(convimg)))


if __name__ == "__main__":
    unittest.main()
