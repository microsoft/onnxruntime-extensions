from PIL import Image
from transformers import AutoProcessor
from onnxruntime_extensions.pp_api import create_processor, load_images, image_pre_process, tensor_result_get_at

import numpy as np


def regen_image(arr):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    # Reverse normalization
    array = arr * std + mean

    # Clip the values to [0, 1] range
    array = np.clip(array, 0, 1)

    # Convert to [0, 255] range and uint8 type
    array = (array * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(array)
    return image


test_image = "test/data/processor/standard_s.jpg"
# test_image = "/temp/passport_s.png"
# test_image = "/temp/passport_s2.png"
model_id = "microsoft/Phi-3-vision-128k-instruct"

processor = create_processor("test/data/processor/phi_3_image.json")
images = load_images([test_image])
c_out = image_pre_process(processor, images)
# print(tensor_result_get_at(c_out, 0))
# print(tensor_result_get_at(c_out, 1))

image = Image.open(test_image)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
messages = [
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"},
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
    {"role": "user", "content": "Provide insightful questions to spark discussion."}
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)


inputs = processor(prompt, [image], return_tensors="pt")
# print(inputs["pixel_values"].numpy())
# print(inputs["image_sizes"])

np.testing.assert_allclose(
    inputs["image_sizes"].numpy(), tensor_result_get_at(c_out, 1))
# np.testing.assert_allclose(inputs["pixel_values"].numpy(), tensor_result_get_at(c_out, 0), rtol=1e-1)
for i in range(17):
    expected = inputs["pixel_values"].numpy()[0, i]
    actual = tensor_result_get_at(c_out, 0)[0, i]
    e_image = regen_image(expected.transpose(1, 2, 0))
    a_image = regen_image(actual.transpose(1, 2, 0))
    e_image.save(f"/temp/e_{i}.png")
    a_image.save(f"/temp/a_{i}.png")

    try:
        np.testing.assert_allclose(inputs["pixel_values"].numpy(
        )[0, i], tensor_result_get_at(c_out, 0)[0, i], rtol=1e-3)
    except AssertionError as e:
        print(str(e))
