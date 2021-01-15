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
|StringSlice | In next version |
|StringLength | In next version |
|StringMapping|  In next version|



### Tokenizer

|**Operator**|**Support State**|
|------------|-----------------|
|GPT2Tokenizer| Supported       |
|BertTokenizer| In next version |
|XLNetTokenizer| In next version |


### Decoder 
|**Operator**|**Support State**|
|------------|-----------------|
|GPT2Decoder | In next version |


## Auxiliary String Operator

[TODO: Add existing operators]

### <a name="StringSlice"></a><a name="StringSlice">**StringSlice**</a>
Do the slice operation to each string element in input tensor. Similar to string slice in python
```python
a = "abcdef"
b = a[1:2]
c = a[3:1:-1]
```
#### Inputs

***data: tensor(string)***
<dd>String tensor to extract slices from.</dd>

***starts: tensor(int64/int32)***
<dd>The tensor of starting indices of corresponding string in data, which has same dimension of data.</dd>

***ends: tensor(int64/int32)***
<dd>The tensor of ending indices of corresponding string in data, which has same dimension of data.</dd>

***steps(optional): tensor(int64/int32)***
<dd>The tensor of slice step of corresponding string in data, which has same dimension of data.</dd>

#### Outputs

***output: tensor(string)***
<dd>Sliced data tensor.</dd>

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
<dd>String tensor to get length of its each string element.</dd>

#### Outputs

***output: tensor(int64)***
<dd>data length tensor.</dd>

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


### <a name="StringMapping"></a><a name="StringMapping">**StringMapping**</a>

StringMapping will map each string element in the input to the corresponding vector according to the mapping file. The mapping file is a utf-8 encoding text file in tsv format:

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

***mapping_file_name***
<dd>the name of your string to vector mapping file.</dd>

***unmapping_value***
<dd>mapping result for unmapped string</dd>

#### Inputs

***data: tensor(string)***
<dd></dd>

#### Outputs

***output: tensor(T)***
<dd>the mapping result of the input</dd>

#### Type Constraints
***T:tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)***
<dd>Constrain input and output types to numerical tensors.</dd>


#### Examples

<details>
<summary>string_mapping</summary>

```python
# what's in vocabulary.txt

# a   0 0 1 2
# b   0 1 2 3
# d   0 1 3 4

node = onnx.helper.make_node(
    'StringMapping',
    inputs=['x'],
    outputs=['y'],
    mapping_file_name='vocabulary.txt',
    unmapping_value=[0,0,0,0]
)


x = ["a", "d", "e"]
y = np.array([[0,0,1,2],[0,1,3,4],[0,0,0,0]], type=np.int64)


expect(node, inputs=[x], outputs=[y],
       name='test_string_mapping')
```
</details>

## Tokenizer

### <a name="GPT2Tokenizer"></a><a name="GPT2Tokenizer">**GPT2Tokenizer**</a>

GPT2Tokenizer that performs byte-level bpe tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/_modules/transformers/tokenization_gpt2.html).

#### Inputs

***data: tensor(string)***
<dd>the string tensor for tokenization</dd>

#### Outputs

***output: tensor(int64)***
<dd>the tokenized result of input</dd>

#### Examples

<details>
<summary>gpt2tokenizer</summary>

```python

node = onnx.helper.make_node(
    'GPT2Tokenizer',
    inputs=['x'],
    outputs=['y'],
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
<dd>the string tensor for tokenization</dd>

#### Outputs

***output: tensor(int64)***
<dd>Tokenized result of the input</dd>

#### Examples

<details>
<summary>word_piece_tokenizer</summary>

```python
```
</details>

### <a name="XLNetTokenizer"></a><a name="XLNetTokenizer">**XLNetTokenizer**</a>

GPT2Tokenizer that performs SentencePiece tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/model_doc/xlnet.html#xlnettokenizer).

#### Inputs

***data: tensor(string)***
<dd>The string tensor for tokenization</dd>

#### Outputs

***output: tensor(int64)***
<dd>Tokenized result of the input</dd>

#### Examples

<details>
<summary>word_piece_tokenizer</summary>

```python

```
</details>

## Decoder

### <a name="GPT2Decoder"></a><a name="GPT2Decoder">**GPT2Decoder**</a>