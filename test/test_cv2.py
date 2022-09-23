import unittest
import numpy as np
from PIL import Image
from onnxruntime_extensions import OrtPyFunction, ONNXRuntimeError, util


class TestOpenCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_image_reader(self):
        img_file = util.get_test_data_file('data', 'pineapple.jpg')

        img_nhwc = None
        # since the ImageReader is not included the offical release due to code compliance issue,
        # it will be test optionally in this case.
        try:
            rdr = OrtPyFunction.from_customop("ImageReader")
            img_nhwc = rdr([img_file])
        except ONNXRuntimeError as e:
            pass

        if img_nhwc is not None:
            self.assertEqual(img_nhwc.shape[0], 1)
            self.assertEqual(img_nhwc.shape[3], 3)
            actual = img_nhwc.squeeze().transpose((2, 0, 1))
            actual = np.stack((actual[2], actual[1], actual[0]))
            actual = actual.transpose((1, 2, 0))

            # crossing check with pillow image API.
            pyimg = Image.open(img_file).convert('RGB')
            expected = np.asarray(pyimg)
            np.testing.assert_array_equal(actual, expected)

    def test_gaussian_blur(self):
        img_file = util.get_test_data_file('data', 'pineapple.jpg')
        img = Image.open(img_file).convert('RGB')
        img_arr = np.asarray(img, dtype=np.float32) / 255.
        img_arr = np.expand_dims(img_arr, 0)
        gb = OrtPyFunction.from_customop('GaussianBlur')
        gb_img = gb(img_arr, np.array([3, 3], dtype=np.int64), np.array([0.0, 0.0]))
        convimg = Image.fromarray((np.squeeze(gb_img, 0) * 255).astype(np.uint8), "RGB")
        # convimg.save('temp_pineapple.jpg')
        self.assertFalse(np.allclose(np.asarray(img), np.asarray(convimg)))

    def test_image_decoder(self):
        input_image_file = util.get_test_data_file("data", "test_colors.jpg")

        model = OrtPyFunction.from_customop("ImageDecoder")
        input_data = open(input_image_file, 'rb').read()
        raw_input_image = np.frombuffer(input_data, dtype=np.uint8)

        actual = model(raw_input_image)
        actual = np.asarray(actual, dtype=np.uint8)
        self.assertEqual(actual.shape[2], 3)

        expected = Image.open(input_image_file).convert('RGB')
        expected = np.asarray(expected, dtype=np.uint8).copy()

        # Convert the image to BGR format since cv2 is default BGR format.
        red = expected[:,:,0].copy()
        expected[:,:,0] = expected[:,:,2].copy()
        expected[:,:,2] = red

        self.assertEqual(actual.shape[0], expected.shape[0])
        self.assertEqual(actual.shape[1], expected.shape[1])
        self.assertEqual(actual.shape[2], expected.shape[2])

        self.assertTrue(np.allclose(actual, expected, atol=1))


if __name__ == "__main__":
    unittest.main()
