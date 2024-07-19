from PIL import Image
from transformers import AutoProcessor
from onnxruntime_extensions.pp_api import create_processor, load_images, image_pre_process, tensor_result_get_at

import numpy as np


test_image = "test/data/processor/passport.png"
model_id = "microsoft/Phi-3-vision-128k-instruct"
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


processor = create_processor("test/data/processor/phi_3_image.json")
images = load_images([test_image])
c_out = image_pre_process(processor, images)
# print(tensor_result_get_at(c_out, 0))
# print(tensor_result_get_at(c_out, 1))

np.testing.assert_allclose(inputs["image_sizes"].numpy(), tensor_result_get_at(c_out, 1))
np.testing.assert_allclose(inputs["pixel_values"].numpy(), tensor_result_get_at(c_out, 0), rtol=1e-1)
