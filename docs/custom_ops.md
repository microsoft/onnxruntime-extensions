# Operators


## Natural language operators

### BertTokenizer

<details>
<summary>BertTokenizer details</summary>

BertTokenizer replicates `encode_plus` function of [BertTokenizer (huggingface version )](https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer).

#### Inputs

***text: tensor(string)*** The string tensor for tokenization

#### Attributes

***vocab_file: string***

The content of vocab which has same with huggingface.

***do_lower_case: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to lowercase the input when tokenizing.

***do_basic_tokenize: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to do basic tokenization before WordPiece.

***unk_token: string***

The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.

***sep_token: string***

The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.

***pad_token: string***

The token used for padding, for example when batching sequences of different lengths.

***cls_token: string***

The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

***mask_token: string***

The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

***tokenize_chinese_chars: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to tokenize Chinese characters.

***strip_accents: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for :obj:`lowercase` (as in the original BERT).

***tokenize_punctuation: int64_t*** (default is 0, 1 represents True, 0 represents False)

Splits punctuation on a piece of text.

***remove_control_chars: int64_t*** (default is 0, 1 represents True, 0 represents False)

Remove control chars(such as NUL, BEL) in the text.

***truncation_strategy_name: string***

The name of truncation strategy, it could be `longest_first`, `only_first`, `only_second`, `longest_from_back`.

#### Outputs

***input_ids: tensor(int64_t)***

List of token ids.

***token_type_ids: tensor(64_t)***

List of token type ids

***attention_mask: tensor(64_t)***

List of indices specifying which tokens should b
e attended to by the model


#### Examples

```python
import transformers

bert_cased_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

node = onnx.helper.make_node(
    'BertTokenizer',
    inputs=['text'],
    outputs=['tokens'],
)

text = "Hello world louder"
inputs = np.array([text], dtype=object),

bert_tokenize_result = bert_cased_tokenizer.tokenize(text)

input_ids = np.array(bert_tokenize_result[0])
token_type_ids = np.array(bert_tokenize_result[1])
attention_mask = np.array(bert_tokenize_result[2])

expect(node, inputs=[inputs],
       outputs=[input_ids, token_type_ids, attention_mask], name='test_bert_tokenizer')
```
</details>

### BertTokenizerDecoder

<details>
<summary>BertTokenizerDecoder details</summary>

BertTokenizerDecoder replicates `decode` function of [BertTokenizer (huggingface version )](https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer).

#### Inputs

***token_ids: tensor(int64)***

List of tokenized input ids.

***indices: tensor(int64)***

List of `[start_position, end_position]` to indicate what segments of input ids should be decoded. This input only enabled when attribute `use_indices`=1.

Usually, it is used to decode the slot in the text.

#### Attributes

***vocab_file: string***

The content of vocab which has same with huggingface.

***unk_token: string***

The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.

***sep_token: string***

The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.

***pad_token: string***

The token used for padding, for example when batching sequences of different lengths.

***cls_token: string***

The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

***mask_token: string***

The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

***suffix_indicator: string***

The suffix indicator.

***use_indices: int64_t***

Whether use second input.

***skip_special_tokens: int64_t***

Whether or not to remove special tokens in the decoding.

***clean_up_tokenization_spaces: int64_t***

Whether or not to clean up the tokenization spaces.

#### Outputs

***sentences: tensor(int64_t)***

The decoded sentences.

#### Examples


```python
import transformers

def get_file_content(path):
  with open(path, "rb") as file:
    return file.read()
  
bert_cased_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
bert_cased_tokenizer.save('.', 'bert')


node = onnx.helper.make_node(
    'BertTokenizerDecoder',
    inputs=['token_ids'],
    outputs=['sentences'],
    vocab_file=get_file_content("bert-vocab.txt")
)

text = "Hello world louder"
token_ids = np.array([bert_cased_tokenizer.tokenize(text)], dtype=object),
sentences = np.array(text)


expect(node, inputs=[token_ids],
       outputs=[sentences], name='test_bert_tokenizer')
```
</details>



### GPT2Tokenizer

<details>
<summary>GPT2Tokenizer details</summary>

GPT2Tokenizer that performs byte-level bpe tokenization to the input tensor, based on the [hugging face version](https://huggingface.co/transformers/_modules/transformers/tokenization_gpt2.html).

#### Attributes

***vocab***

The **content** of the vocabulary file, its format is same with [hugging face](https://huggingface.co/gpt2/resolve/main/vocab.json).

***merges***

The **content** of the merges file, its format is same with [hugging face](https://huggingface.co/gpt2/resolve/main/merges.txt).

***padding_length(optional)***

When the input is a set of query, the tokenized result is ragged tensor, so we need to pad the tensor to tidy tensor and the `padding_length` indicates the strategy of the padding. When the padding_length equals -1, we will pad the tensor to length of longest row. When the padding_length is more than 0, we will pad the tensor to the number of padding_length.

The default value of `padding_length` is -1.

#### Inputs

***data: tensor(string)***

The string tensor for tokenization

#### Outputs

***input_ids: tensor(int64)***

The tokenized ids of input

***attention_mask: tensor(int64)***

A tensor indicates which part of input_ids is padded.

#### Examples


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

### WordpieceTokenizer

<details>
<summary>WordpieceTokenizer details</summary>


WordpieceTokenizer that performs WordPiece tokenization to the input tensor,
based on the [hugging face version](https://huggingface.co/transformers/model_doc/bert.html#WordpieceTokenizer).
[WordpieceTokenizer](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/WordpieceTokenizer.md)
from *tensorflow_text* can be implemented by a pair of nodes
*RegexSplitWithOffets* followed by *WordpieceTokenizer*.
it 

#### Attributes

***vocab***

The **content** of the vocabulary file, its format is same with
[hugging face](https://huggingface.co/gpt2/resolve/main/vocab.json).

***suffix_indicator***

Suffix added to token not in the first position before looking into the vocabulary.

***unk_token***

Unknown tokens. Every token not found in the vocabulary is replaced by this one.

***max_input_chars_per_word***

Maximum number of characters per token (optional, defaults to 200).

#### Inputs

***data: tensor(string)***

The string tensor for tokenization

***row_indices: tensor(int64)*** Empty or the fndices of every first token of input sentences.
`indices[i+1] - indices[i]` is the number of tokens in input `i`.

[WordpieceTokenizer](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/WordpieceTokenizer.md)
includes two steps. The first one splits sentences into words and then splits
every work into tokens. This operator only implements the second step.
The first one can be done with operator *StringRegexSplit*.
This parameter can either be empty or it can be the third output
of operator *StringRegexSplit*.

#### Outputs

***tokens: tensor(string)*** Every token.

***token_indices: tensor(int32)*** Indices of each token. -1 means a token outside the vocabulary.

***row_indices: tensor(int64)*** Indices of every first token of input sentences.
`indices[i+1] - indices[i]` is the number of tokens in input `i`.
These are updates row indices given as inputs or new ones if the second input is empty.

#### Examples


```python
words = ["want", "##want",
         "##ed", "wa", "un", "runn", "##ing"]
vocab = {w: i + 10 for i, w in enumerate(words)}
st = json.dumps(vocab)
nodes = []
mkv = helper.make_tensor_value_info
reg = helper.make_tensor(
    "pattern", onnx_proto.TensorProto.STRING, [1, ], ["(\\s)".encode('ascii')])
reg_empty = helper.make_tensor(
    "keep_pattern", onnx_proto.TensorProto.STRING, [0, ], [])

nodes = [
    helper.make_node(
        'StringRegexSplitWithOffsets,
        inputs=['text', 'pattern', 'keep_pattern'],
        outputs=['words', 'begin_end', 'indices'],
        name='StringRegexPlsitOpName',
        domain='ai.onnx.contrib'),
    helper.make_node(
        'WordpieceTokenizer',
        inputs=['words', 'indices'],
        outputs=['out0', 'out1', 'out2'],
        name='WordpieceTokenizerOpName',
        domain='ai.onnx.contrib',
        vocab=st.encode('utf-8'),
        suffix_indicator="##",
        unk_token="[UNK]")
]
inputs = [mkv('text', onnx_proto.TensorProto.STRING, [None])]
graph = helper.make_graph(
    nodes, 'test0', inputs, [
        mkv('out0', onnx_proto.TensorProto.STRING, [None]),
        mkv('out1', onnx_proto.TensorProto.INT32, [None]),
        mkv('out2', onnx_proto.TensorProto.INT64, [None]),
        mkv('words', onnx_proto.TensorProto.STRING, [None]),
        mkv('indices', onnx_proto.TensorProto.INT64, [None])],
    [reg, reg_empty])
model = helper.make_model(
    graph, opset_imports=[helper.make_operatorsetid(domain, 1)])

text = np.array(["unwanted running", "unwantedX running"], dtype=object)
tokens = np.array(['un', '##want', '##ed', 'runn', '##ing', 'un', '##want', '##ed',
                  '[UNK]', 'runn', '##ing'], dtype=object),
indices = np.array([14, 11, 12, 15, 16, 14, 11, 12, -1, 15, 16], dtype=int32)
row_indices = np.array([ 0,  5, 11], dtype=int64)

expect(model, inputs=[text], outputs=[tokens, indices, row_indices],
       name='test_bert_tokenizer')
```

</details>

### SentencepieceTokenizer

<details>
<summary>SentencepieceTokenizer details</summary>

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

inputs = np.array(["Hello world", "Hello world louder"], dtype=object),
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


### BasicTokenizer

<details>
<summary>BasicTokenizer details</summary>

TODO: is this still supported?

BasicTokenizer performs basic tokenization to input string tensor, based on [basic tokenizer in BertTokenizer(hugging face version)](https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer).

#### Inputs

***text: tensor(string)*** The string tensor for tokenization

#### Attributes

***do_lower_case: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to lowercase the input when tokenizing.

***tokenize_chinese_chars: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to tokenize Chinese characters.

***strip_accents: int64_t*** (default is 1, 1 represents True, 0 represents False)

Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for :obj:`lowercase` (as in the original BERT).

***tokenize_punctuation: int64_t*** (default is 0, 1 represents True, 0 represents False)

Splits punctuation on a piece of text.

***remove_control_chars: int64_t*** (default is 0, 1 represents True, 0 represents False)

Remove control chars(such as NUL, BEL) in the text.

#### Outputs

***tokens: tensor(string)*** Tokenized tokens.

#### Examples

```python
import transformers

tokenizer = transformers.BasicTokenizer()

node = onnx.helper.make_node(
    'BasicTokenizer',
    inputs=['text'],
    outputs=['tokens'],
)

inputs = np.array([ "Hello world louder"], dtype=object),
tokens = np.array(tokenizer(inputs), dtype=int32)

expect(node, inputs=[inputs],
       outputs=[tokens], name='test_basic_tokenizer')
```
</details>


### BlingFireSentenceBreaker

TODO

### BpeTokenizer

TODO


## String operators

### StringEqual

<details>
<summary>StringEqual details</summary>

Compares two strings and returns true if they are equal and false if not.

#### Inputs

***x: tensor(string)***

The first string input

***x: tensor(string)***

The second string input

#### Outputs

***z: tensor(boolean)***

String with replacements.

</details>


### StringHash

<details>
<summary>StringHash details</summary>


Hashes the input string based on the number of buckets

#### Inputs

***input: tensor(string)***

The string to hash

***num_buckets: tensor(int64)***

The number of buckets (must be equal to 1?)

#### Outputs

***name: tensor(int64)***

The hash value of the string

</details>


### StringHashFast

<details>
<summary>StringHashFast details</summary>


A faster implementation of StringHash.

</details>


### StringJoin  

<details>
<summary>StringJoin details</summary>


Join an array of strings

#### Inputs

***input_X: tensor(string)***

The input array of strings

***input_sep: tensor(string)***

The string separator for the resulting joing

***input_axis: tensor(int64)***

The axis along which to joing

#### Outputs

***out: tensor(string)***

The resulting joined string

#### Examples


```bash

input_X = [["a", "b", "c"], ["aa", "bb", ""]]
input_sep=";"
input_axis = 1

out = ["a;b;c", "aa;bb;"]

input_axis = 0

out = ['a;aa', 'b;bb', 'c;']


</details>


### StringRegexReplace

<details>
<summary>StringRegexReplace details</summary>


String replacement based on [Re2-format](https://github.com/google/re2/wiki/Syntax) regular expressions.

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

### StringECMARegexReplace

<details>
<summary>StringECMARegexReplace details</summary>

String replacement based on [ECMA-format](https://en.cppreference.com/w/cpp/regex/ecmascript) regular expressions.

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


***ignore_case: int64*** (default is 0)

Replace 

#### Outputs

***output: tensor(string)***

String with replacements.

#### Examples


```python

node = onnx.helper.make_node(
    'StringRegexReplace',
    inputs=['text', 'pattern', 'rewrite'],
    outputs=['y'],
)

text = np.array([['def myfunc():'], ['def dummy():']])
pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
rewrite = np.array([r'static PyObject* py_$1(void) {'])
y = [['static PyObject* py_myfunc(void) {'],
     ['static PyObject* py_dummy(void) {']]

expect(node, inputs=[text, pattern, rewrite], outputs=[y],
       name='test_string_regex_replace')
```

</details>



### StringSplit 

TODO

### StringUpper  

TODO

### StringLower

TODO

### StringLength

<details>
<summary>StringECMARegexReplace details</summary>

Get the length of each string element in input tensor. Similar to the function `len("abcde"")` in python.

#### Inputs 

***data: tensor(string)***

String tensor to get length of its each string element.

#### Outputs

***output: tensor(int64)***

Data length tensor.

#### Examples


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
 
### StringConcat 

<details>
<summary>StringConcat details</summary>

Concat the corresponding string in the two string tensor. Two input tensors should have the same dimension.

```python
  output = []
  shape = input1.shape
  input1 = input1.flatten()
  input2 = input2.flatten()
  for i in range(len(input1)):
      output.append(input1[i] + input2[i])
  output = np.array(output).reshape(shape)
```

#### Inputs

***input_1: tensor(string)***

The first string tensor.

***input_2: tensor(string)***

The second string tensor.


#### Outputs

***output: tensor(string)***

The result.

#### Examples


```python

node = onnx.helper.make_node(
    'StringConcat',
    inputs=['x', 'y'],
    outputs=['result'],
)

x = np.array(["abcd", "efgh"])
y = np.array(["wxyz", "stuv"])
result = np.array([x[0] + y[0], x[1] + y[1]])

expect(node, inputs=[x, y], outputs=[result],
       name='test_string_concat')
```

</details>

### StringRegexSplitWithOffsets

<details>
<summary>StringRegexSplitWithOffsets details</summary>

Splits string based on regular expressions.

#### Inputs

***text: tensor(string)***

String tensor to extract slices from.

***delim_regex_pattern: tensor(string)***

Splitting attern of the regular expression.

***keep_delim_regex_pattern: tensor(string)***

By default, delimiters are not included in the split string results. Delimiters may be included by specifying a regex pattern keep_delim_regex_pattern.

#### Outputs

***words: tensor(string)*** Tensor of words.

***offsets: tensor(int64)*** 2D tensor with 3 columns:
sentence index, position of the first character, position of the last one (excluded)

***row_indices: tensor(int64)*** Indices of every first token of input sentences.
`row_indices[i+1] - row_indices[i]` is the number of tokens in input `i`.
These are updates row indices given as inputs or new ones if the second input is empty.


#### Examples


```python

node = onnx.helper.make_node(
    'StringRegexSplit',
    inputs=['text', 'pattern', 'rewrite'],
    outputs=['y', 'begin_end', 'indices'],
)

text = np.array(["hello there"])
pattern = np.array([r'\s'])
rewrite = np.array([r'\s'])
y = np.array(["hello", " ", "there"])
z1 = np.array([[0, 0, 5],
               [0, 5, 6],
               [0, 6, 11]], dtype=np.int64)
z2 = np.array([0, 2], dtype=np.int64)

expect(node, inputs=[text, pattern, rewrite], outputs=[y, z1, z2],
       name='test_string_regex_replace')
```

</details>


### StringECMARegexSplitWithOffsets

TODO

### VectorToString

<details>
<summary>VectorToString details</summary>

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

***data: tensor(T)***

Input tensor

#### Outputs

***output: tensor(string)***

The mapping result of the input

#### Type Constraints
***T:tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)***

Constrain input and output types to numerical tensors.


#### Examples


```python
mapping_table = \
  """
  a   0 0 1 2
  b   0 1 2 3
  d   0 1 3 4
  """

node = onnx.helper.make_node(
    'VectorToString',
    inputs=['x'],
    outputs=['y'],
    map=mapping_table,
    unk="unknown_word"
)


x = np.array([[0,0,1,2],[0,1,3,4],[0,0,0,0]], type=np.int64)
y = ["a", "d", "unknown_word"]


expect(node, inputs=[x], outputs=[y],
       name='test_vector_to_string')
```
</details>


### StringToVector

<details>
<summary>StringToVector details</summary>

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

Input tensor

#### Outputs

***output: tensor(T)***

The mapping result of the input

#### Type Constraints
***T:tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(bfloat16), tensor(float16), tensor(float), tensor(double), tensor(bool)***

Constrain input and output types to numerical tensors.

#### Examples


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



### StringSlice 

<details>
<summary>StringSlice details</summary>

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


```python

node = onnx.helper.make_node(
    'StringSlice',
    inputs=['x', 'starts', 'ends', 'steps'],
    outputs=['y'],
)

x = np.array(["abcdef", "hijkl"])
y = np.array([x[0][1:3:1], x[1][3:1:-1]])
starts = np.array([1, 3], dtype=np.int64)
ends = np.array([3, 1], dtype=np.int64)
axes = np.array([0, 1], dtype=np.int64)
steps = np.array([1, 1], dtype=np.int64)

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_string_slice')
```

</details>


### MaskedFill

<details>
<summary>MaskedFill details</summary>


Fills elements of self tensor with value where mask is True. The operator is similar with [`Tensor.masked_fill_`](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_) in pytorch.


#### Inputs

***value: tensor(string)***

The value to fill in with, currently we only support string type and vector&scalar dimension.

***mask: tensor(bool)***

The boolean mask, the dimension of mask tensor should be same with value.

#### Outputs

***output: tensor(string)***

The filled output of input tensor.


#### Examples


```python

node = onnx.helper.make_node(
    'MaskedFill',
    inputs=['value', 'mask'],
    outputs=['output']
)


value = np.array(["a", "b", "c", "d"])
mask = np.array([True, False, True, False], dtype=bool)
output = np.array(["a", "c"])


expect(node, inputs=[value, mask], outputs=[output],
       name='test_masked_fill')
```
</details>


### StringRaggedTensorToDense

TODO

### StringMapping

TODO

## Math operators


### Inverse

TODO 

### NegPos

TODO

### SegmentExtraction

TODO

### SegmentSum

TODO

## Tensor operators

### RaggedTensorToSparse

TODO

### RaggedTensorToDense

TODO

### Template

<details>
<summary>Template details</summary>

Description

#### Inputs

***name: tensor(type)***

Description

#### Outputs

***name: tensor(type)***

Description

#### Examples


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


## Azure operators

### OpenAIAudioToText

<details>
<summary>OpenAIAudioToText details</summary>


OpenAIAudioToText operator talks to [openAI audio](https://platform.openai.com/docs/api-reference/audio) endpoints.


#### Attributes

***model_uri:string***

Endpoint uri, like "https://api.openai.com/v1/audio/transcriptions".

***audio_format:string***

The format of the audio, by default "wav".

#### Inputs

***auth_token: tensor(string)***

An access token comes with openAI subscription.

***model_name: tensor(string)***

Model name to send to the endpoint, such as "whisper-1".

***response_format: tensor(string)***

Expected format of the response, either be "text" or "json".

***audio_blob: tensor(uint8)***

A byte array containing raw data from the audio file.

#### Outputs

***transcriptions: tensor(string)***


#### Examples


```python

import os
import numpy as np

from onnx import *
from onnxruntime_extensions import PyOrtFunction, util, get_library_path
from onnxruntime import *


def create_openai_audio_model():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [1])
    model = helper.make_tensor_value_info('model_name', TensorProto.STRING, [1])
    response_format = helper.make_tensor_value_info('response_format', TensorProto.STRING, [-1])
    file = helper.make_tensor_value_info('file', TensorProto.UINT8, [-1])
    transcriptions = helper.make_tensor_value_info('transcriptions', TensorProto.STRING, [-1])

    invoker = helper.make_node('OpenAIAudioToText',
                               ['auth_token', 'model_name', 'response_format', 'file'],
                               ['transcriptions'],
                               domain='com.microsoft.extensions',
                               name='audio_invoker',
                               model_uri='https://api.openai.com/v1/audio/transcriptions',
                               audio_format='wav')

    graph = helper.make_graph([invoker], 'graph', [auth_token, model, response_format, file], [transcriptions])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])

    onnx.save(model, 'openai_audio.onnx')


create_openai_audio_model()
opt = SessionOptions()
opt.register_custom_ops_library(get_library_path())
sess = InferenceSession(os.path.join(test_data_dir, "openai_audio.onnx"),
                        opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
auth_token = np.array([os.getenv('MYAUTH', '')])
model = np.array(['whisper-1'])
response_format = np.array(['text'])

with open(os.path.join(test_data_dir, "test16.wav"), "rb") as _f:
    audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
    ort_inputs = {
        "auth_token": auth_token,
        "model_name": model,
        "response_format": response_format,
        "file": audio_blob,
    }
    out = sess.run(None, ort_inputs)[0]
```
</details>


### AzureTextToText

<details>
<summary>AzureTextToText details</summary>


AzureTextToText talks to a GPT model hosted by [Azure openAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/).


#### Attributes

***model_uri:string***

Endpoint uri, like "https://myname-aoai-test.openai.azure.com/openai/deployments/mydeploy/chat/completions?api-version=2023-05-15'".

#### Inputs

***auth_token: tensor(string)***

An access token comes with Azure openAI subscription.

***chat: tensor(string)***

A json string in requested [format](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=rest-api).

#### Outputs

***response_format: tensor(string)***

A json string as response.


#### Examples


```python

import os
import numpy as np

from onnx import *
from onnxruntime_extensions import PyOrtFunction, util, get_library_path
from onnxruntime import *


def create_azure_chat_model():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    chat = helper.make_tensor_value_info('chat', TensorProto.STRING, [-1])
    response = helper.make_tensor_value_info('response', TensorProto.STRING, [-1])

    invoker = helper.make_node('AzureTextToText', ['auth_token', 'chat'], ['response'],
                               domain='com.microsoft.extensions',
                               name='chat_invoker',
                               model_uri='https://rashuai-aoai-test.openai.azure.com/openai/deployments/randysgpt/chat/completions?api-version=2023-05-15')

    graph = helper.make_graph([invoker], 'graph', [auth_token, chat], [response])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])

    onnx.save(model, 'azure_chat.onnx')


create_azure_chat_model()
opt = SessionOptions()
opt.register_custom_ops_library(get_library_path())
sess = InferenceSession(os.path.join(test_data_dir, "azure_chat.onnx"), opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
auth_token = np.array([os.getenv('MYAUTH', '')])
chat = np.array([r'{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},{"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},{"role": "user", "content": "Do other Azure AI services support this too?"}]}'])
ort_inputs = {
    "auth_token": auth_token,
    "chat": chat,
}
out = sess.run(None, ort_inputs)[0]
```
</details>


### AzureTritonInvoker

<details>
<summary>AzureTritonInvoker details</summary>


AzureTritonInvoker talks to [Azure Machine Learning triton services](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?view=azureml-api-2&tabs=azure-cli%2Cendpoint).


#### Attributes

***model_uri:string***

Endpoint uri, like "'https://endpoint-12345678.westus.inference.ml.azure.com".

***model_name:string***

***model_version:string***

A version string, like "1", or "2".

#### Inputs

***auth_token: tensor(string)***

An access token comes with Azure Machine Learning model deployment.

***inputs: tensor(variadic)***

Tensors of any supported onnx data type.

#### Outputs

***outputs: tensor(variadic)***

Tensors of any supported onnx data type.


#### Examples


```python

import os
import numpy as np

from onnx import *
from onnxruntime_extensions import PyOrtFunction, util, get_library_path
from onnxruntime import *


def createAddf():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-1.westus2.inference.ml.azure.com',
                               model_name='addf', model_version='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addf.onnx')


def createAddf8():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.DOUBLE, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.DOUBLE, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.DOUBLE, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-2.westus2.inference.ml.azure.com',
                               model_name='addf8', model_version='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addf8.onnx')


def createAddi4():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.INT32, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.INT32, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-3.westus2.inference.ml.azure.com',
                               model_name='addi4', model_version='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addi4.onnx')


def createAnd():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.BOOL, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.BOOL, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.BOOL, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-4.westus2.inference.ml.azure.com',
                               model_name='and', model_version='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_and.onnx')


def createStr():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    str_in = helper.make_tensor_value_info('str_in', TensorProto.STRING, [-1])
    str_out1 = helper.make_tensor_value_info('str_out1', TensorProto.STRING, [-1])
    str_out2 = helper.make_tensor_value_info('str_out2', TensorProto.STRING, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'str_in'], ['str_out1','str_out2'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-5.westus2.inference.ml.azure.com',
                               model_name='str', model_version='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, str_in], [str_out1, str_out2])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_str.onnx')


def run_add_f():
    opt = SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    sess = InferenceSession(os.path.join(test_data_dir, "triton_addf.onnx"),
                            opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
    auth_token = np.array([os.getenv('MYAUTH', '')])
    x = np.array([1,2,3,4]).astype(np.float32)
    y = np.array([4,3,2,1]).astype(np.float32)
    ort_inputs = {
        "auth_token": auth_token,
        "X": x,
        "Y": y
    }
    out = sess.run(None, ort_inputs)[0]


def run_add_f8():
    opt = SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    sess = InferenceSession(os.path.join(test_data_dir, "triton_addf8.onnx"),
                            opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
    auth_token = np.array([os.getenv('MYAUTH', '')])
    x = np.array([1,2,3,4]).astype(np.double)
    y = np.array([4,3,2,1]).astype(np.double)
    ort_inputs = {
        "auth_token": auth_token,
        "X": x,
        "Y": y
    }
    out = sess.run(None, ort_inputs)[0]


def run_add_i4():
    opt = SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    sess = InferenceSession(os.path.join(test_data_dir, "triton_addi4.onnx"),
                            opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
    auth_token = np.array([os.getenv('MYAUTH', '')])
    x = np.array([1,2,3,4]).astype(np.int32)
    y = np.array([4,3,2,1]).astype(np.int32)
    ort_inputs = {
        "auth_token": auth_token,
        "X": x,
        "Y": y
    }
    out = sess.run(None, ort_inputs)[0]


def run_and():
    opt = SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    sess = InferenceSession(os.path.join(test_data_dir, "triton_and.onnx"),
                            opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
    auth_token = np.array([os.getenv('MYAUTH', '')])
    x = np.array([True, True])
    y = np.array([True, False])
    ort_inputs = {
        "auth_token": auth_token,
        "X": x,
        "Y": y
    }
    out = sess.run(None, ort_inputs)[0]


def run_str():
    opt = SessionOptions()
    opt.register_custom_ops_library(get_library_path())
    sess = InferenceSession(os.path.join(test_data_dir, "triton_str.onnx"),
                            self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
    auth_token = np.array([os.getenv('MYAUTH', '')])
    str_in = np.array(['this is the input'])
    ort_inputs = {
        "auth_token": auth_token,
        "str_in": str_in
    }
    outs = sess.run(None, ort_inputs)
```
</details>