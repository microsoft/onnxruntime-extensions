## Operator Schemas

### Auxiliary String Operator

|**Operator**|**Support State**|
|------------|-----------------|
|StringEqual |  Supported        |
|StringHash  |  Supported        |
|StringToHashBucketFast|Supported|
|StringJoin  | Supported         |
|StringRegexReplace| Supported  |
|StringSplit | Supported       |
|StringUpper  | Supported     |
|StringSlice | Under development|
|StringLength | Under development |
|StringToVector|  Under development|
|VectorToString| Under development |

### Tokenizer

|**Operator**|**Support State**|
|------------|-----------------|
|GPT2Tokenizer| Supported       |
|BertTokenizer| Under development |
|XLNetTokenizer| Under development |
|SentencepieceTokenizer| Supported       |

## Auxiliary String Operator

[TODO: Add existing operators]

### <a name="StringRegexReplace"></a><a name="StringRegexReplace">**StringRegexReplace**</a>

String replacement based on regular expressions.

#### Inputs

***text: tensor(string)***

String tensor to extract slices from.

***pattern: tensor(string)***

Pattern of the regular expression.

***rewrite: tensor(string)***

Replacement.

#### Attributes

***global_replace: int64*** (default is 1)

Replace all strings matching the pattern or the first one.

#### Outputs

***output: tensor(string)***

String with replacements.

#### Examples

<details>
<summary>string_slice</summary>

```python

node = onnx.helper.make_node(
    'StringRegexReplace',
    inputs=['text', 'pattern', 'rewrite'],
    outputs=['y'],
)

text = np.array([['def myfunc():'], ['def dummy():']])
pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
rewrite = np.array([r'static PyObject* py_\1(void) {'])
y = [['static PyObject* py_myfunc(void) {'],
     ['static PyObject* py_dummy(void) {']]

expect(node, inputs=[text, pattern, rewrite], outputs=[y],
       name='test_string_regex_replace')
```

</details>

### <a name="StringSlice"></a><a name="StringSlice">**StringSlice**</a>

Do the slice operation to each string element in input tensor. Similar to string slice in python

```python
a = "abcdef"
b = a[1:2]
c = a[3:1:-1]
```

#### Inputs

***data: tensor(string)***

String tensor to extract slices from.

***starts: tensor(int64/int32)***

The tensor of starting indices of corresponding string in data, which has same dimension of data.

***ends: tensor(int64/int32)***

The tensor of ending indices of corresponding string in data, which has same dimension of data.

***steps(optional): tensor(int64/int32)***

The tensor of slice step of corresponding string in data, which has same dimension of data.If steps is empty tensor, we will use default value 1 for each string

#### Outputs

***output: tensor(string)***

Sliced data tensor.

#### Examples

<details>
<summary>string_slice</summary>

```python

node = onnx.helper.make_node(
    'StringSlice',
    inputs=['x', 'starts', 'ends', 'steps'],
    outputs=['y'],
)

x = ["abcdef", "hijkl"]
y = [x[0][1:3:1], x[1][3:1:-1]]
starts = np.array([1, 3], dtype=np.int64)
ends = np.array([3, 1], dtype=np.int64)
axes = np.array([0, 1], dtype=np.int64)
steps = np.array([1, 1], dtype=np.int64)

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_string_slice')
```

</details>

### <a name="StringLength"></a><a name="StringLength">**StringLength**</a>

Get the length of each string element in input tensor. Similar to the function `len("abcde"")` in python.

#### Inputs 

***data: tensor(string)***

String tensor to get length of its each string element.

#### Outputs

***output: tensor(int64)***

Data length tensor.

#### Examples

<details>
<summary>string_length</summary>

```python

node = onnx.helper.make_node(
    'StringLength',
    inputs=['x'],
    outputs=['y']
)

x = ["abcdef", "hijkl"]
y = np.array([len(x[0]), len(x[1])], dtype=np.int64)


expect(node, inputs=[x], outputs=[y],
       name='test_string_length')
```
</details>


### <a name="StringToVector"></a><a name="StringToVector">**StringToVector**</a>

StringToVector will map each string element in the input to the corresponding vector according to the mapping file. The mapping file is a utf-8 encoding text file in tsv format:

    <string>\t<scalar_1>\s<scalar_2>\s<scalar_3>...<scalar_n>

Unmapped string will output the value of the attribute `unmapping_value`.

Example:

*Attributes:*

- `mapping_file_name`: vocabulary.txt
  ```
  a   0 0 1 2
  b   0 1 2 3
  d   0 1 3 4
  ```
  
- `unmapping_value`: [0 0 0 0]

*Inputs:*
- data: ["a", "d", "e"]

*Ouputs:*
- output: [[0,0,1,2],[0,1,3,4],[0,0,0,0]]

#### Attributes

***mapping_file_name:string***

The name of your string to vector mapping file.

***unmapping_value:list(int)***

Mapping result for unmapped string

#### Inputs

***data: tensor(string)***

Iut tensor

#### Outputs

***output: tensor(T)***

The mapping result of the input

#### Type Constraints
***T:tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)***

Constrain input and output types to numerical tensors.

#### Examples

<details>
<summary>string_to_vector</summary>

```python
# what's in vocabulary.txt

mapping_table = \
"""
a   0 0 1 2
b   0 1 2 3
d   0 1 3 4
"""

node = onnx.helper.make_node(
    'StringToVector',
    inputs=['x'],
    outputs=['y'],
    mapping_table=mapping_table,
    unmapping_value=[0,0,0,0]
)


x = ["a", "d", "e"]
y = np.array([[0,0,1,2],[0,1,3,4],[0,0,0,0]], type=np.int64)


expect(node, inputs=[x], outputs=[y],
       name='test_string_to_vector')
```

</details>

### <a name="VectorToString"></a><a name="VectorToString">**VectorToString**</a>

VectorToString is the contrary operation to the `StringToVector` , they share same format of mapping table:

    <string>\t<scalar_1>\s<scalar_2>\s<scalar_3>...<scalar_n>

Unmapped vector will output the value of the attribute `unk`.

Example:

*Attributes:*

- `map`: 
  ```
  a   0 0 1 2
  b   0 1 2 3
  d   0 1 3 4
  ```

- `unk`: "unknown_word"

*Inputs:*
- data: [[0,0,1,2],[0,1,3,4],[0,0,0,0]]

*Ouputs:*
- output: ["a", "d", "unknown_word" ]

#### Attributes

***mapping_file_name***

the formative mapping table

***unmapping_value***

the result returned when a vector aren't found in the map

#### Inputs

***data: tensor(string)***

Input tensor

#### Outputs

***output: tensor(T)***

The mapping result of the input

#### Type Constraints
***T:tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)***

Constrain input and output types to numerical tensors.


#### Examples

<details>
<summary>vector_to_string</summary>

```python
mapping_table = \
  """
  a   0 0 1 2
  b   0 1 2 3
  d   0 1 3 4
  """

node = onnx.helper.make_node(
    'StringToVector',
    inputs=['x'],
    outputs=['y'],
    map=mapping_table,
    unk="unknown_word"
)


x = np.array([[0,0,1,2],[0,1,3,4],[0,0,0,0]], type=np.int64)
y = ["a", "d", "unknown_worde"]


expect(node, inputs=[x], outputs=[y],
       name='test_vector_to_string')
```
</details>

## Tokenizer

### <a name="GPT2Tokenizer"></a><a name="GPT2Tokenizer">**GPT2Tokenizer**</a>

GPT2Tokenizer that performs byte-level bpe tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/_modules/transformers/tokenization_gpt2.html).

#### Attributes

***vocab***

The **content** of the vocabulary file, its format is same with [hugging face](https://huggingface.co/gpt2/resolve/main/vocab.json).

***merges***

The **content** of the merges file, its format is same with [hugging face](https://huggingface.co/gpt2/resolve/main/merges.txt).


#### Inputs

***data: tensor(string)***

The string tensor for tokenization

#### Outputs

***output: tensor(int64)***

The tokenized result of input

#### Examples

<details>
<summary>gpt2tokenizer</summary>

```python
def get_file_content(path):
  with open(path, "rb") as file:
    return file.read()

node = onnx.helper.make_node(
    'GPT2Tokenizer',
    inputs=['x'],
    outputs=['y'],
    vocab=get_file_content(vocabulary_file),
    merges=get_file_content(merges_file)
)

x = ["hey cortana"]
y = np.array([20342, 12794, 2271], dtype=np.int64)

expect(node, inputs=[x], outputs=[y],
       name='test_gpt2_tokenizer')
```
</details>


### <a name="BertTokenizer"></a><a name="BertTokenizer">**BertTokenizer**</a>

BertTokenizer that performs WordPiece tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer).

#### Inputs

***data: tensor(string)***

The string tensor for tokenization

#### Outputs

***output: tensor(int64)***

Tokenized result of the input

#### Examples

<details>
<summary>word_piece_tokenizer</summary>

```python
```
</details>

### <a name="SentencepieceTokenizer"></a><a name="SentencepieceTokenizer">**SentencepieceTokenizer**</a>

SentencepieceTokenizer replicates [SentencepieceTokenizer](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/SentencepieceTokenizer.md).

#### Inputs

***data: tensor(string)*** The string tensor for tokenization

***nbest_size: tensor(int64)***	A scalar for sampling. nbest_size = {0,1}: No sampling is performed.
(default) nbest_size > 1: samples from the nbest_size results. nbest_size < 0: assuming that
nbest_size is infinite and samples from the all hypothesis (lattice) using
forward-filtering-and-backward-sampling algorithm.

***alpha: tensor(float)*** A scalar for a smoothing parameter. Inverse temperature for probability rescaling.

***reverse: tensor(bool)*** Reverses the tokenized sequence (Default = false)

***add_bos: tensor(bool)*** Add beginning of sentence token to the result (Default = false)

***add_eos: tensor(bool)*** Add end of sentence token to the result (Default = false).
When reverse=True beginning/end of sentence tokens are added after reversing.

#### Attributes

***model: string*** The sentencepiece model serialized proto as stored as a string.

#### Outputs

***tokens: tensor(int32)*** Indices of each token.

***indices: tensor(int64)*** Indices of every first token of input sentences.
`indices[i+1] - indices[i]` is the number of tokens in input `i`.

Tokenized result of the input

#### Examples

<details>
<summary>example 1</summary>

```python

url = "https://github.com/microsoft/ort-customops/raw/main/test/data/test_sentencepiece_ops_model__6.txt"
with urllib.request.urlopen(url) as f:
    content = f.read()
model = np.array(list(base64.decodebytes(content.encode())), dtype=np.uint8)

node = onnx.helper.make_node(
    'SentencepieceTokenizer',
    inputs=['inputs', 'nbest_size', 'alpha', 'add_bos', 'add_eos', 'reverse'],
    outputs=['indices', 'output'],
    mapping_file_name='vocabulary.txt',
    unmapping_value="unknown_word",
    model=model
)

inputs = np.array(["Hello world", "Hello world louder"], dtype=np.object),
nbest_size = np.array([0], dtype=np.float32),
alpha = np.array([0], dtype=np.float32),
add_bos = np.array([0], dtype=np.bool_),
add_eos = np.array([0], dtype=np.bool_),
reverse = np.array([0], dtype=np.bool_)

tokens = array([17486,  1017, 17486,  1017,   155, 21869], dtype=int32)
indices = array([0, 2, 6], dtype=int64)

expect(node, inputs=[inputs, nbest_size, alpha, add_bos, add_eos, reverse],
       outputs=[tokens, indices], name='sp')
```
</details>

### <a name="XLNetTokenizer"></a><a name="XLNetTokenizer">**XLNetTokenizer**</a>

GPT2Tokenizer that performs SentencePiece tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/model_doc/xlnet.html#xlnettokenizer).

#### Inputs

***data: tensor(string)***
The string tensor for tokenization

#### Outputs

***output: tensor(int64)***

Tokenized result of the input

#### Examples

<details>
<summary>word_piece_tokenizer</summary>

```python

```
</details>
