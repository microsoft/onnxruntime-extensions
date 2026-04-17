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

bert_cased_tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-cased')

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
  
bert_cased_tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-cased')
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
    model=model,
    domain='ai.onnx.contrib'
)

inputs = np.array(["Hello world", "Hello world louder"], dtype=object),
nbest_size = np.array([0], dtype=np.float32),
alpha = np.array([0], dtype=np.float32),
add_bos = np.array([0], dtype=np.bool_),
add_eos = np.array([0], dtype=np.bool_),
reverse = np.array([0], dtype=np.bool_)

tokens = np.array([17486,  1017, 17486,  1017,   155, 21869], dtype=np.int32)
indices = np.array([0, 2, 6], dtype=np.int64)

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


### CLIPTokenizer

<details>
<summary>CLIPTokenizer details</summary>

Byte-pair-encoding (BPE) tokenizer matching the CLIP text encoder from HuggingFace/OpenAI. Converts input strings into token id sequences.

#### Attributes

***vocab: string***

JSON vocabulary mapping tokens to ids (contents of `vocab.json`).

***merges: string***

Merge rules (contents of `merges.txt`).

***padding_length: int64_t*** (default is -1)

If positive, the output is right-padded (or truncated) to this length. When -1 no padding is performed and outputs stay ragged.

#### Inputs

***input: tensor(string)***

1D string tensor containing the input texts.

#### Outputs

***input_ids: tensor(int64)***

Tensor of token ids.

***attention_mask: tensor(int64)***

Mask with the same shape as `input_ids` (1 for real tokens, 0 for padding).

***offset_mapping: tensor(int64)*** (optional)

If requested, per-token `(begin, end)` byte offsets into the corresponding input string.

</details>


### RobertaTokenizer

<details>
<summary>RobertaTokenizer details</summary>

BPE tokenizer compatible with HuggingFace's RoBERTa tokenizer. Uses the same attributes and I/O contract as `CLIPTokenizer`.

#### Attributes

***vocab: string***

JSON vocabulary (contents of `vocab.json`).

***merges: string***

BPE merge rules (contents of `merges.txt`).

***padding_length: int64_t*** (default is -1)

Optional fixed output length. See `CLIPTokenizer`.

#### Inputs

***input: tensor(string)***

1D string tensor of input texts.

#### Outputs

***input_ids: tensor(int64)***

Token ids.

***attention_mask: tensor(int64)***

Attention mask, same shape as `input_ids`.

***offset_mapping: tensor(int64)*** (optional)

Per-token byte offsets into each input string.

</details>


### SpmTokenizer

<details>
<summary>SpmTokenizer details</summary>

SentencePiece-compatible tokenizer built on top of the shared BPE kernel. Produces tokens equivalent to HuggingFace's "fast" SentencePiece tokenizers (e.g. Llama, T5, XLM-RoBERTa).

#### Attributes

***vocab: string***

JSON vocabulary produced from a SentencePiece model.

***merges: string***

SentencePiece merge rules.

***padding_length: int64_t*** (default is -1)

Optional fixed output length.

#### Inputs

***input: tensor(string)***

1D string tensor of inputs.

#### Outputs

***input_ids: tensor(int64)***

Tensor of token ids.

***attention_mask: tensor(int64)***

Attention mask with the same shape as `input_ids`.

***offset_mapping: tensor(int64)*** (optional)

Per-token byte offsets.

</details>


### HfBertTokenizer

<details>
<summary>HfBertTokenizer details</summary>

HuggingFace-compatible BERT WordPiece tokenizer. Behaves like `BertTokenizer`'s `__call__` method but with a smaller attribute surface. Produces ids, attention masks and token type ids in a single op.

#### Attributes

***vocab_file: string***

Contents of `vocab.txt`.

***do_lower_case: int64_t*** (default is 1)

Lowercase inputs before tokenization.

***strip_accents: int64_t*** (default is 1)

Strip accents as part of normalization.

#### Inputs

***input: tensor(string)***

1D string tensor containing the texts to tokenize.

#### Outputs

***input_ids: tensor(int64)***

Token ids.

***attention_mask: tensor(int64)***

Attention mask, same shape as `input_ids`.

***token_type_ids: tensor(int64)*** (optional)

Segment ids. All zero for single-sentence input.

</details>


### HfJsonTokenizer

<details>
<summary>HfJsonTokenizer details</summary>

Loads a HuggingFace `tokenizer.json` directly and dispatches to the appropriate kernel (BPE or Unigram). Matches HuggingFace fast tokenizers at inference time.

#### Attributes

***tokenizer_config: string***

Contents of `tokenizer.json` (and optionally `tokenizer_config.json`).

***tokenizer_vocab: string*** (optional)

Additional vocabulary data when the tokenizer uses an external vocab file.

#### Inputs

***input: tensor(string)***

1D string tensor of inputs.

#### Outputs

***input_ids: tensor(int64)***

Token ids.

***attention_mask: tensor(int64)***

Attention mask matching `input_ids`.

***offset_mapping: tensor(int64)*** (optional)

Per-token byte offsets.

</details>


### SentencepieceDecoder

<details>
<summary>SentencepieceDecoder details</summary>

Decodes a sequence of SentencePiece ids back into a string.

#### Attributes

***model: string***

Serialized SentencePiece model (`*.model`).

#### Inputs

***ids: tensor(int64)***

1D or 2D tensor of ids. When 2D the leading dimension must be 1.

***fairseq: tensor(bool)*** (optional)

Scalar flag. When true the `fairseq` vocab-id offset convention is applied.

#### Outputs

***output: tensor(string)***

Scalar string containing the decoded text.

</details>


### BpeDecoder

<details>
<summary>BpeDecoder details</summary>

Decodes BPE token ids (GPT-2 / CLIP / RoBERTa style) back into text.

#### Attributes

***id_vocab: string***

Newline-separated token strings indexed by id.

***byte_decoder: string***

Reverse byte-to-unicode mapping used by GPT-2 BPE encoders.

***added_tokens: string*** (optional)

Extra tokens appended to the base vocabulary.

***all_special_ids: string*** (optional)

Comma-separated list of special token ids.

***skip_special_tokens: int64_t*** (default is 0)

When 1, ids in `all_special_ids` are skipped during decoding.

***en_normalization: int64_t*** (default is 0)

Apply a minimal English-oriented post-processing step (e.g. undo leading-space markers).

***whitespace_token: string*** (optional)
***bos_token: string*** (optional)
***eos_token: string*** (optional)
***unk_token: string*** (optional)

Optional overrides for well-known special tokens.

#### Inputs

***ids: tensor(int64)***

1D or 2D tensor of token ids.

#### Outputs

***output: tensor(string)***

Decoded string tensor.

</details>


### TrieTokenizer

<details>
<summary>TrieTokenizer details</summary>

Trie-based longest-match tokenizer used by RWKV-style models.

#### Attributes

***vocab: string***

Newline-separated vocab where each line has the form `index token length`. `token` is a Python-repr-encoded byte string.

#### Inputs

***input: tensor(string)***

1D string tensor of inputs.

#### Outputs

***output: tensor(int64)***

2D right-padded tensor of token ids; padding uses id `0`.

</details>


### TrieDetokenizer

<details>
<summary>TrieDetokenizer details</summary>

Inverse of `TrieTokenizer`. Converts 1D or 2D id tensors back to strings using the same trie vocabulary.

#### Attributes

***vocab: string***

Same vocabulary format as `TrieTokenizer`.

#### Inputs

***ids: tensor(int64)***

1D or 2D tensor of token ids.

#### Outputs

***output: tensor(string)***

Decoded text, one string per row.

</details>


### BlingFireSentenceBreaker

<details>
<summary>BlingFireSentenceBreaker details</summary>

Segments an input string into sentences using a compiled [BlingFire](https://github.com/microsoft/BlingFire) model.

#### Attributes

***model: string***

Raw bytes of the compiled BlingFire sentence-breaking model (`*.bin`).

***max_sentence: int64_t*** (default is -1)

If positive, limits the number of returned sentences.

#### Inputs

***input: tensor(string)***

Scalar input string.

#### Outputs

***output: tensor(string)***

1D tensor of sentences.

</details>


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


A faster implementation of StringHash. Computes hash values for each input string modulo `num_buckets`.

#### Inputs

***input: tensor(string)***

The strings to hash.

***num_buckets: tensor(int64)***

The number of hash buckets (scalar). Each output value will be in the range `[0, num_buckets)`.

#### Outputs

***output: tensor(int64)***

The hashed values, with the same shape as `input`.

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

<details>
<summary>StringSplit details</summary>

Splits each string in the input by a separator, producing a ragged (sparse) representation of the resulting tokens.

#### Inputs

***input: tensor(string)***

1D string tensor to split.

***sep: tensor(string)***

Scalar string separator used to split each element of `input`. If empty, the string is split on whitespace.

***skip_empty: tensor(bool)***

Scalar boolean. When true, empty substrings are removed from the output.

#### Outputs

***indices: tensor(int64)***

2D tensor of shape `[N, 2]` containing `(row, col)` coordinates of each output token in the ragged representation.

***values: tensor(string)***

1D tensor of `N` tokens produced by splitting, in row-major order.

***shape: tensor(int64)***

2-element tensor describing the dense shape `[num_rows, max_row_width]` of the ragged tensor.

</details>

### StringUpper

<details>
<summary>StringUpper details</summary>

Converts every ASCII character in each string of the input tensor to uppercase. Non-ASCII bytes are left unchanged.

#### Inputs

***input: tensor(string)***

String tensor of arbitrary shape.

#### Outputs

***output: tensor(string)***

String tensor of the same shape as `input` with uppercased strings.

</details>

### StringLower

<details>
<summary>StringLower details</summary>

Converts each string in the input tensor to lowercase using Unicode case folding.

#### Inputs

***input: tensor(string)***

String tensor of arbitrary shape.

#### Outputs

***output: tensor(string)***

String tensor of the same shape as `input` with lowercased strings.

</details>

### StringStrip

<details>
<summary>StringStrip details</summary>

Removes leading and trailing whitespace characters from every string in the input tensor. Similar to `str.strip()` in Python.

#### Inputs

***input: tensor(string)***

String tensor of arbitrary shape.

#### Outputs

***output: tensor(string)***

String tensor of the same shape as `input` with whitespace stripped.

</details>

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

<details>
<summary>StringECMARegexSplitWithOffsets details</summary>

Splits strings using a regular expression in the ECMAScript dialect and reports the byte offsets of every produced token. Provides the same functionality as `StringRegexSplitWithOffsets` but uses `std::regex` instead of `re2`, allowing ECMAScript regex features.

#### Inputs

***input: tensor(string)***

String tensor to split.

***pattern: tensor(string)***

Scalar string containing the ECMAScript regex splitting pattern.

***keep_pattern: tensor(string)***

Scalar string. Delimiter matches that also match this pattern are preserved as tokens in the output. Pass an empty string to drop all delimiters.

#### Attributes

***ignore_case: int64_t*** (default is 0)

When set to 1 the regex is matched case-insensitively.

#### Outputs

***words: tensor(string)***

1D tensor containing the split tokens.

***offsets: tensor(int64)***

2D tensor of shape `[num_tokens, 3]` where each row is `(sentence_index, begin_byte, end_byte)`.

***row_indices: tensor(int64)***

1D tensor of row offsets such that tokens of the i-th input string occupy `[row_indices[i], row_indices[i+1])` in `words`.

</details>

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

<details>
<summary>StringRaggedTensorToDense details</summary>

Converts a ragged string tensor to a dense 2D string tensor, padding shorter rows with a fill value.

#### Inputs

***row_splits: tensor(int64)***

1D tensor with the starting position of each row in `values`. Row `i` contains `values[row_splits[i]:row_splits[i+1]]`.

***values: tensor(string)***

1D flat string tensor holding the concatenated row values.

***default_value_shape: tensor(int64)***

1D tensor describing the target dense shape. Only used to determine the number of columns.

***default_value: tensor(string)***

Scalar string used to pad rows that are shorter than the longest row.

#### Outputs

***output: tensor(string)***

2D dense string tensor with padding applied.

</details>

### StringMapping

<details>
<summary>StringMapping details</summary>

Maps each element of the input string tensor to another string using a user-supplied dictionary. Strings not found in the dictionary are passed through unchanged.

#### Attributes

***map: string***

A string containing one mapping per line. Each line has the form `key\tvalue`, where key and value are separated by a tab character.

#### Inputs

***input: tensor(string)***

Input string tensor of arbitrary shape.

#### Outputs

***output: tensor(string)***

Output string tensor of the same shape as `input` after mapping.

</details>

## Math operators


### Inverse

<details>
<summary>Inverse details</summary>

Computes the matrix inverse of a 2D floating-point tensor.

#### Inputs

***input: tensor(float)***

A 2D square matrix of shape `[N, N]`.

#### Outputs

***output: tensor(float)***

The inverse of the input matrix, of shape `[N, N]`.

</details>

### NegPos

<details>
<summary>NegPos details</summary>

Splits an input tensor into its negative and positive parts. Equivalent to `min(x, 0)` and `max(x, 0)` returned separately.

#### Inputs

***input: tensor(float)***

Input tensor of arbitrary shape.

#### Outputs

***neg: tensor(float)***

Tensor with the same shape as `input`; contains `x` where `x < 0`, else `0`.

***pos: tensor(float)***

Tensor with the same shape as `input`; contains `x` where `x >= 0`, else `0`.

</details>

### SegmentExtraction

<details>
<summary>SegmentExtraction details</summary>

Extracts contiguous non-zero segments from a 1D integer input. For every maximal run of non-zero values, the start and end positions are returned.

#### Inputs

***input: tensor(int64)***

1D input tensor.

#### Outputs

***position: tensor(int64)***

2D tensor of shape `[num_segments, 2]` where each row is `(begin, end)` (end exclusive).

***value: tensor(int64)***

1D tensor of length `num_segments` with the value inside each segment.

</details>

### SegmentSum

<details>
<summary>SegmentSum details</summary>

Computes sums along segments of the first axis of a tensor, similar to TensorFlow's `tf.math.segment_sum`.

#### Inputs

***data: tensor(float)***

The values to reduce. The first dimension is the segment axis.

***segment_ids: tensor(int64)***

1D tensor with the same length as `data.shape[0]`. Must be non-decreasing.

#### Outputs

***output: tensor(float)***

Tensor where `output[i]` is the sum of all rows of `data` whose corresponding `segment_ids` equal `i`.

</details>

### StftNorm

<details>
<summary>StftNorm details</summary>

Computes a short-time Fourier transform (STFT) of a 1D signal and returns the magnitude spectrogram. The implementation uses a Hann-style sliding window.

#### Attributes

***onesided: int64_t*** (default is 1)

If 1, only the non-redundant positive-frequency half of the spectrum is returned (length `n_fft / 2 + 1`). If 0, the full spectrum is returned.

#### Inputs

***pcm: tensor(float)***

1D audio signal.

***n_fft: tensor(int64)***

Scalar FFT size.

***hop_length: tensor(int64)***

Scalar hop length between consecutive frames.

***window: tensor(float)***

1D window function of length `frame_length`.

***frame_length: tensor(int64)***

Scalar frame length (must equal `n_fft`).

#### Outputs

***output: tensor(float)***

3D tensor of shape `[1, num_frames, num_freq_bins]` containing the magnitude spectrogram.

</details>

### SplitSignalSegments

<details>
<summary>SplitSignalSegments details</summary>

Partitions an audio signal into segments of voiced/high-energy regions based on a simple short-time energy threshold.

#### Inputs

***input: tensor(float)***

1D audio signal.

***sr: tensor(int64)***

Scalar sample rate in Hz.

***frame_ms: tensor(int64)***

Scalar analysis frame length in milliseconds.

***hop_ms: tensor(int64)***

Scalar hop length between analysis frames in milliseconds.

***energy_threshold_db: tensor(float)***

Scalar energy threshold in dBFS. Frames with average energy below this are treated as silence.

#### Outputs

***segments: tensor(int64)***

2D tensor of shape `[num_segments, 2]` where each row contains the `(begin_sample, end_sample)` indices of a detected segment.

</details>

### MergeSignalSegments

<details>
<summary>MergeSignalSegments details</summary>

Merges adjacent audio segments whose gap is shorter than a configurable threshold. Typically used as a post-processing step after `SplitSignalSegments`.

#### Inputs

***segments: tensor(int64)***

2D tensor of shape `[N, 2]` with `(begin, end)` indices, as produced by `SplitSignalSegments`.

***merge_gap_ms: tensor(int64)***

Scalar gap threshold in milliseconds. Segments separated by less than this value are merged.

#### Outputs

***output: tensor(int64)***

2D tensor of shape `[M, 2]` (M <= N) of the merged segment boundaries.

</details>

## Tensor operators

### RaggedTensorToSparse

<details>
<summary>RaggedTensorToSparse details</summary>

Converts a ragged tensor's row lengths to a COO-style sparse indexing representation.

#### Inputs

***n_element: tensor(int64)***

1D tensor holding the number of elements in each row.

#### Outputs

***output_0: tensor(int64)***

2D tensor of `(row, col)` indices for every element.

***output_1: tensor(int64)***

1D tensor of length 2 containing the dense shape `[num_rows, max_row_width]`.

</details>

### RaggedTensorToDense

<details>
<summary>RaggedTensorToDense details</summary>

Converts a ragged int64 tensor to a dense 2D tensor, padding shorter rows with a configurable value.

#### Attributes

***missing_value: int64_t*** (default is -1)

Value used to pad short rows.

#### Inputs

***input0: tensor(int64)***

1D row-splits tensor indicating the start index of each row within `input3`.

***input1: tensor(int64)***

1D tensor of flat indices (unused by some consumers; reserved).

***input2: tensor(int64)***

1D tensor of length 2 describing the target dense shape `[num_rows, max_row_width]`.

***input3: tensor(int64)***

1D flat values tensor.

#### Outputs

***output: tensor(int64)***

2D dense tensor with missing elements filled by `missing_value`.

</details>

## Audio operators

### AudioDecoder

<details>
<summary>AudioDecoder details</summary>

Decodes a byte stream containing an encoded audio file (WAV, MP3, or FLAC) into a float PCM tensor. Optionally resamples the audio to a target sample rate.

#### Attributes

***downsampling_rate: int64_t*** (default is 0)

Target sample rate. When 0 the native sample rate of the decoded stream is used.

***stereo_to_mono: int64_t*** (default is 1)

If 1, multi-channel audio is mixed down to a single mono channel.

***target_sample_rate: int64_t*** (default is 0)

Alias for `downsampling_rate`; when non-zero the decoded audio is resampled to this rate.

#### Inputs

***input: tensor(uint8)***

1D tensor of raw bytes representing the encoded audio file.

***format: tensor(string)*** (optional)

Scalar describing the container format. Accepted values: `"wav"`, `"mp3"`, `"flac"`. When absent the format is detected from the file header.

#### Outputs

***output: tensor(float)***

2D tensor of shape `[1, num_samples]` with the decoded (and optionally resampled) PCM samples in the range `[-1, 1]`.

</details>

## Vision operators

### DecodeImage

<details>
<summary>DecodeImage details</summary>

Decodes an encoded image (PNG, JPEG, BMP, TIFF, …) into an `HxWx3` uint8 tensor.

#### Attributes

***color_space: string*** (default is "BGR")

Color ordering of the output. Valid values are `"RGB"` and `"BGR"`.

#### Inputs

***input: tensor(uint8)***

1D tensor containing the raw encoded image bytes.

#### Outputs

***output: tensor(uint8)***

3D tensor of shape `[H, W, 3]`.

</details>

### EncodeImage

<details>
<summary>EncodeImage details</summary>

Encodes a 3-channel `HxWx3` uint8 image tensor to PNG or JPEG bytes.

#### Attributes

***format: string*** (default is "png")

Output image format. Valid values are `"png"` and `"jpg"` (or `"jpeg"`).

#### Inputs

***input: tensor(uint8)***

3D tensor of shape `[H, W, 3]` in BGR order.

#### Outputs

***output: tensor(uint8)***

1D tensor of encoded image bytes.

</details>

### DrawBoundingBoxes

<details>
<summary>DrawBoundingBoxes details</summary>

Draws bounding boxes on a BGR image tensor.

#### Attributes

***thickness: int64_t*** (default is 4)

Line thickness of the drawn rectangles, in pixels.

***num_classes: int64_t*** (default is 10)

Number of class colors to cycle through.

***mode: string*** (default is "XYXY")

Interpretation of the box coordinates. One of `"XYXY"`, `"XYWH"`, or `"CENTER_XYWH"`.

***colour_by_classes: int64_t*** (default is 1)

When 1, boxes of the same class share a colour. When 0, each box gets a unique colour from the palette.

#### Inputs

***image: tensor(uint8)***

3D tensor of shape `[H, W, 3]` in BGR order.

***boxes: tensor(float)***

2D tensor of shape `[N, 6]`. Each row is `(class_id, score, x0, y0, x1, y1)` (or equivalent depending on `mode`).

#### Outputs

***output: tensor(uint8)***

Image tensor with boxes drawn, same shape as `image`.

</details>

### GaussianBlur

<details>
<summary>GaussianBlur details</summary>

Applies a 2D Gaussian blur to an image tensor using OpenCV's `cv::GaussianBlur`.

#### Inputs

***input: tensor(float)***

4D image tensor of shape `[N, H, W, C]`.

***ksize: tensor(int64)***

1D tensor of length 2 specifying the kernel size `[kx, ky]` (odd positive integers).

***sigma: tensor(double)***

1D tensor of length 2 specifying the Gaussian standard deviation along X and Y.

#### Outputs

***output: tensor(float)***

Blurred tensor with the same shape as `input`.

</details>

### ImageDecoder

<details>
<summary>ImageDecoder details</summary>

Decodes raw encoded image bytes using OpenCV's `cv::imdecode`. Similar to `DecodeImage` but always returns BGR and does not expose a color-space attribute.

#### Inputs

***input: tensor(uint8)***

1D tensor of encoded image bytes.

#### Outputs

***output: tensor(uint8)***

3D tensor of shape `[H, W, C]` containing the decoded BGR image.

</details>

### ImageReader

<details>
<summary>ImageReader details</summary>

Reads an image from a file path using OpenCV's `cv::imread` and returns the decoded tensor.

#### Inputs

***input: tensor(string)***

Scalar string with the path of the image file to read.

#### Outputs

***output: tensor(uint8)***

4D tensor of shape `[1, H, W, C]` containing the decoded BGR image.

</details>

## CUDA operators

The following operators execute on CUDA devices only. They are only registered when the library is built with `USE_CUDA`. Unless otherwise noted each op supports `float`, `float16` (`MFloat16`), and in some cases `bfloat16` (`BFloat16`).

### FastGelu

<details>
<summary>FastGelu details</summary>

Fused CUDA kernel computing `gelu(x + bias)` using the fast tanh-based approximation.

#### Inputs

***input: tensor(T)***

Input tensor of any shape. `T` is one of `float`, `float16`, `bfloat16`.

***bias: tensor(T)*** (optional)

Bias added elementwise before applying Gelu. Broadcast to the shape of `input`.

#### Outputs

***output: tensor(T)***

Same shape as `input`.

</details>

### MulSigmoid

<details>
<summary>MulSigmoid details</summary>

Computes `x * sigmoid(x)` (the SiLU / Swish activation) in a single fused CUDA kernel.

#### Inputs

***input: tensor(T)***

Input tensor. `T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***output: tensor(T)***

Same shape as `input`.

</details>

### MulMulSigmoid

<details>
<summary>MulMulSigmoid details</summary>

Computes `x * y * sigmoid(y)` in a single fused CUDA kernel. Tensors must have the same shape.

#### Inputs

***x: tensor(T)***, ***y: tensor(T)***

`T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***output: tensor(T)***

Tensor with the same shape as the inputs.

</details>

### NegXPlus1

<details>
<summary>NegXPlus1 details</summary>

Computes `1 - x` elementwise on CUDA.

#### Inputs

***input: tensor(T)***

`T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***output: tensor(T)***

Same shape as `input`.

</details>

### ReplaceZero

<details>
<summary>ReplaceZero details</summary>

Replaces every zero element of the input with a scalar value.

#### Attributes

***by: float*** (default is 0.0)

Replacement value for zero entries.

#### Inputs

***input: tensor(T)***

`T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***output: tensor(T)***

Same shape as `input`.

</details>

### AddSharedInput

<details>
<summary>AddSharedInput details</summary>

Computes `A + B` and `A + C` in one kernel launch, sharing the read of `A`.

#### Inputs

***A: tensor(T)***, ***B: tensor(T)***, ***C: tensor(T)***

`T` is one of `float`, `float16`, `bfloat16`. `B` and `C` must have the same shape as `A`.

#### Outputs

***AB: tensor(T)***, ***AC: tensor(T)***

Elementwise sums `A + B` and `A + C`.

</details>

### MulSharedInput

<details>
<summary>MulSharedInput details</summary>

Computes `A * B` and `A * C` in one kernel launch, sharing the read of `A`.

#### Inputs

***A: tensor(T)***, ***B: tensor(T)***, ***C: tensor(T)***

`T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***AB: tensor(T)***, ***AC: tensor(T)***

Elementwise products `A * B` and `A * C`.

</details>

### ScatterNDOfShape

<details>
<summary>ScatterNDOfShape details</summary>

Allocates a zero tensor of the given shape and applies a `ScatterND` reduction. Equivalent to `ScatterND(ConstantOfShape(shape, 0), indices, updates, reduction=...)` but fused.

#### Attributes

***reduction: string*** (default is "add")

Reduction to apply to scattered updates. One of `"add"`, `"mul"`, `"min"`, `"max"`.

#### Inputs

***shape: tensor(int64)***

1D tensor describing the output shape. Must live on CPU.

***indices: tensor(int64)***

Indices into the output, as in standard ScatterND.

***updates: tensor(T)***

Values to scatter. `T` is one of `float`, `float16`, `bfloat16`.

#### Outputs

***output: tensor(T)***

Tensor of the requested shape with updates applied.

</details>

### MaskedScatterNDOfShape

<details>
<summary>MaskedScatterNDOfShape details</summary>

Variant of `ScatterNDOfShape` that ignores entries of `indices` equal to a configurable mask value.

#### Attributes

***reduction: string*** (default is "add")

Same as `ScatterNDOfShape`.

***maskedValue: int64_t***

Index value that causes the corresponding update to be skipped.

#### Inputs

Same as `ScatterNDOfShape`.

#### Outputs

Same as `ScatterNDOfShape`.

</details>

### Transpose2DCastFP16

<details>
<summary>Transpose2DCastFP16 details</summary>

Fused 2D transpose + cast from `float` to `float16`.

#### Inputs

***input: tensor(float)***

2D tensor of shape `[M, N]`.

#### Outputs

***output: tensor(float16)***

2D tensor of shape `[N, M]`.

</details>

### Transpose2DCastFP32

<details>
<summary>Transpose2DCastFP32 details</summary>

Fused 2D transpose + cast from `float16` to `float`.

#### Inputs

***input: tensor(float16)***

2D tensor of shape `[M, N]`.

#### Outputs

***output: tensor(float)***

2D tensor of shape `[N, M]`.

</details>

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

