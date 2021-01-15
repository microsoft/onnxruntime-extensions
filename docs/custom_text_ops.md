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
|GPT2Tokenizer| Supported      |
|WordPieceTokenizer| In next version |
|BertTokenizer | In next version |


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
       name='test_slice')
```
</details>

### <a name="StringLength"></a><a name="StringLength">**StringLength**</a>

Get the length of each string element in input tensor. Similar to the function `len("abcde"")` in python.

#### Inputs 

***data: tensor(string)***
<dd>String tensor to get length of its each string element.</dd>

#### Outputs

***output: tensor(int)***
<dd>data length tensor.</dd>

#### Examples

<details>
<summary>string_slice</summary>

```python

node = onnx.helper.make_node(
    'StringSlice',
    inputs=['x'],
    outputs=['y'],
)

x = ["abcdef", "hijkl"]
y = [len(x[0]), len(x[1]) ]


expect(node, inputs=[x], outputs=[y],
       name='test_slice')
```
</details>


### <a name="StringMapping"></a><a name="StringMapping">**StringMapping**</a>

#### Attributes

***mapping_file_name***
<dd>the name of your string to vector mapping file.</dd>

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
<summary>string_slice</summary>

```python

node = onnx.helper.make_node(
    'StringSlice',
    inputs=['x'],
    outputs=['y'],
)

x = ["abcdef", "hijkl"]
y = [len(x[0]), len(x[1]) ]


expect(node, inputs=[x], outputs=[y],
       name='test_slice')

```
</details>

## Tokenizer

### <a name="GPT2Tokenizer"></a><a name="GPT2Tokenizer">**GPT2Tokenizer**</a>

### <a name="WordPieceTokenizer"></a><a name="WordPieceTokenizer">**WordPieceTokenizer**</a>

### <a name="BertTokenizer"></a><a name="BertTokenizer">**BertTokenizer**</a>


## Decoder

### <a name="GPT2Decoder"></a><a name="GPT2Decoder">**GPT2Decoder**</a>