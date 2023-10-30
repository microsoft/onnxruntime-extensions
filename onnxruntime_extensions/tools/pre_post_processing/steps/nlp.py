# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from collections import OrderedDict
from pathlib import Path

from typing import Optional, Union, Dict
from ..step import Step


class TokenizerParam(object):
    def __init__(self, vocab_or_file: Union[Path, dict], **kwargs):
        self.vocab_or_file = vocab_or_file
        self.tweaked_bos_id = 1
        self.strip_accents = 0
        self.do_lower_case = 0
        self.is_sentence_pair = 0
        self.__assigned_with_kwargs(**kwargs)

    def __assigned_with_kwargs(self, **kwargs):
        for key in self.__dict__.keys():
            if key in kwargs and kwargs.get(key) is not None:
                setattr(self, key, kwargs[key])


class SentencePieceTokenizer(Step):
    def __init__(
        self,
        tokenizer_param: TokenizerParam,
        nbest_size=0,
        alpha=1.0,
        reverse=False,
        add_bos=False,
        add_eos=False,
        name: Optional[str] = None,
    ):
        """
        Brief:
            SentencePieceTokenizer has actually 6 inputs in definition, but we allow user to provide only text input,
            and make the others, "nbest_size", "alpha", "add_bos", "add_eos", "reverse" optional.
        Args:
            tokenizer_param: some essential infos to build a tokenizer
            you can create a TokenizerParam object like:
                tokenizer_param = TokenizerParam(vocab_size=tokenizer.vocab_size,
                                                 tweaked_bos_id=tokenizer.tweaked_bos_id)

            nbest_size: int, optional (default = 0)
            alpha: float, optional (default = 1.0)
            reverse: bool, optional (default = False)
            add_bos: bool, optional (default = False)
            add_eos: bool, optional (default = False)
                  Please see more detail explanation in 
                  https://www.tensorflow.org/text/api_docs/python/text/SentencepieceTokenizer#args

            name: Optional name of step. Defaults to 'SentencePieceTokenizer'

        """
        super().__init__(
            ["input_text", "nbest_size", "alpha", "add_bos", "add_eos", "reverse"], ["input_ids", "attention_mask"], name
        )
        self._tokenizer_param = tokenizer_param
        # python bool value (True/False) is not supported in c++, so we use 0/1 to represent bool
        self._optional_kwargs = dict(
            nbest_size=nbest_size, alpha=alpha, add_bos=int(add_bos), add_eos=int(add_eos), reverse=int(reverse)
        )

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        # input text
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        input_shape_0 = input_shape_str0.split(",")
        # ideally, we should support batch input, each batch has different length and output a token
        # !!! But, the implementation of SentencePieceTokenizer is not batch supported, inputs will be flatten to 1D
        # in the sentence-piece kernel
        assert input_type_str0 == "string"

        # we have to do this hack here, because some models tweaked bos_id to 0, but we have still 1
        # as default value in model file.
        # it is only a temporary solution, we will remove it in the future.
        tweak_bos_id = False
        if self._tokenizer_param.tweaked_bos_id != 1 and self._optional_kwargs["add_bos"]:
            self._optional_kwargs["add_bos"] = 0
            tweak_bos_id = True

        batch_dim = input_shape_0[0] if len(input_shape_0) > 1 else "1"
        prefix_ = f'step_{self.step_num}'
        output_shape_str = f"{batch_dim}, {prefix_}__num_ids"

        def build_input_declare():
            input_base = f"{input_type_str0}[{input_shape_str0}] {self.input_names[0]}"
            return input_base

        def build_call_para():
            para_base = ["input_with_batch"]
            para_base.append("i64_nbest_size")
            para_base.append("f32_alpha")
            para_base.append("bool_add_bos")
            para_base.append("bool_add_eos")
            para_base.append("bool_reverse")
            return ",".join(para_base)

        def build_forward_declare():
            # default values for nbest_size, alpha, add_bos, add_eos, reverse
            declare_base = [
                f"i64_nbest_size = Constant <value = int64[1] {{{self._optional_kwargs['nbest_size']}}}> ()",
                f"f32_alpha = Constant <value = float[1] {{ {self._optional_kwargs['alpha']} }}> ()",
                f"bool_add_bos = Constant <value = bool[1] {{{self._optional_kwargs['add_bos']}}}> ()",
                f"bool_add_eos = Constant <value = bool[1] {{{self._optional_kwargs['add_eos']}}}> ()",
                f"bool_reverse = Constant <value = bool[1] {{{self._optional_kwargs['reverse']}}}> ()",
            ]

            return "\n".join(declare_base)

        # TODO Camembert and XLMRoberta tokenizers has a different bos_token_id (0) from the default value (1)
        # Now, we are hacking it.

        def hack_bos_id():
            if tweak_bos_id:
                return f'''
                k_start = Constant <value = int32[1] {{{self._tokenizer_param.tweaked_bos_id}}}> ()
                input_ids_concat02 = Concat <axis = 0> (k_start, token)
                input_ids_bdim = Unsqueeze(input_ids_concat02, i64_0)
                '''
            else:
                return '''
                input_ids_bdim = Unsqueeze(token, i64_0)
                '''

        def build_unsqueeze():
            if len(input_shape_0) == 1:
                return f"""
            input_with_batch = Unsqueeze({self.input_names[0]}, i64_0)
            """
            else:
                return f"""
            input_with_batch = Identity({self.input_names[0]})
            """

        converter_graph = onnx.parser.parse_graph(
            f"""\
            SentencePiecetokenizer ({build_input_declare()}) 
                => (int64[{output_shape_str}] {self.output_names[0]},int64[{output_shape_str}] {self.output_names[1]})  
            {{
                {build_forward_declare()}
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()
                i64_0 = Constant <value = int64[1] {{0}}> ()
                {build_unsqueeze()}
                token,idx =  com.microsoft.extensions.SentencepieceTokenizer ({build_call_para()})
                {hack_bos_id()}
                {self.output_names[0]} = Cast <to = 7> (input_ids_bdim)
                attention_mask_i32=Greater({self.output_names[0]}, i64_neg1)
                {self.output_names[1]} = Cast <to = 7> (attention_mask_i32)
            }}
            """
        )

        with open(self._tokenizer_param.vocab_or_file, "rb") as f:
            content = f.read()

        token_model_attr = onnx.helper.make_attribute("model", content)
        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == "SentencepieceTokenizer")
        converter_graph.node[node_idx].attribute.append(token_model_attr)

        return converter_graph


def _vocab_to_dict(vocab_or_file: Union[Dict[str, int], Path, str]):
    if isinstance(vocab_or_file, (Path, str)):
        # read from file
        import json
        with open(vocab_or_file, "r") as f:
            vocab = json.load(f)
    else:
        vocab = vocab_or_file

    ordered_vocab = OrderedDict(sorted(vocab.items(), key=lambda item: int(item[1])))

    vocab = "\n".join(ordered_vocab.keys())
    return dict(vocab_file=vocab)


class BertTokenizer(Step):
    def __init__(self, tokenizer_param: TokenizerParam, need_token_type_ids_output: bool = False, name: Optional[str] = None):
        """
        Brief: This step is used to convert the input text into the input_ids, attention_mask, token_type_ids.
            It supports an input of a single string for classification models, or two strings for QA models.
        Args:
            tokenizer_param: some essential infos to build a tokenizer,
            You can create a TokenizerParam like this:
                tokenizer_param = TokenizerParam(vocab=tokenizer.vocab, # vocab is dict or file_path
                                                 strip_accents = True or False (Optional),
                                                 do_lower_case = True or False (Optional)
                                                 )

            name: Optional name of step. Defaults to 'BertTokenizer'
            need_token_type_ids_output: last outout `token_type_ids` is not required in some Bert based models. (e.g. DistilBert, etc.) can optionally
                                        choose to add it in graph for step.

        """
        outputs = []
        if need_token_type_ids_output:
            outputs.extend(["input_ids", "attention_mask"])
        else:
            outputs.extend(["input_ids", "attention_mask", "token_type_ids"])
        super().__init__(["input_text"], outputs, name)
        self._tokenizer_param = tokenizer_param

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)

        input_shape_0 = input_shape_str0.split(",")
        prefix_ = f'step_{self.step_num}'
        # only support bath size 1 until tokenizer op supports batch size > 1
        batch_dim = input_shape_0[0] if len(input_shape_0) > 1 else "1"
        output_shape_str = f"{batch_dim}, _{prefix_}__num_ids"
        assert input_type_str0 == "string"

        onnx_tokenizer_impl = "HfBertTokenizer" if self._tokenizer_param.is_sentence_pair else "BertTokenizer"

        def build_output_declare():
            output_base = []
            for out in self.output_names:
                output_base.append(f"int64[{output_shape_str}] {out}")

            return ",".join(output_base)

        def get_tokenizer_ret():
            if onnx_tokenizer_impl == "HfBertTokenizer":
                return ",".join(self.output_names)
            # different output orders for BertTokenizer and HfBertTokenizer
            return "ids,types,mask"

        def build_output_imp():
            if onnx_tokenizer_impl == "HfBertTokenizer":
                return ""

            # BertTokenizer has different output dimensions
            ret_vars = get_tokenizer_ret().split(",")
            ret_vars[1], ret_vars[2] = ret_vars[2], ret_vars[1]
            output_str = []

            for idx, out in enumerate(self.output_names):
                output_str.append(f"{out} = Unsqueeze({ret_vars[idx]}, i64_0)")

            return "\n".join(output_str)

        def build_input_declare():
            inputs = f"{input_type_str0}[{input_shape_str0}] {self.input_names[0]}"
            return inputs

        def build_unsqueeze():
            if len(input_shape_0) == 1:
                return f"""            
            input_with_batch = Unsqueeze({self.input_names[0]}, i64_0)
            """
            else:
                return f"""
            input_with_batch = Identity({self.input_names[0]})
            """

        converter_graph = onnx.parser.parse_graph(
            f"""\
            {onnx_tokenizer_impl} ({build_input_declare()}) 
                => ({build_output_declare()})
            {{
                i64_0 = Constant <value = int64[1] {{0}}> ()
                {build_unsqueeze()}
                {get_tokenizer_ret()} = com.microsoft.extensions.{onnx_tokenizer_impl} (input_with_batch)
                {build_output_imp()}
            }}
            """
        )

        bert_tokenizer_param = self._tokenizer_param
        token_model_attr = []

        attrs = _vocab_to_dict(bert_tokenizer_param.vocab_or_file)
        attrs["strip_accents"] = bert_tokenizer_param.strip_accents
        attrs["do_lower_case"] = bert_tokenizer_param.do_lower_case

        for attr in attrs:
            token_model_attr.append(onnx.helper.make_attribute(attr, attrs[attr]))

        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == onnx_tokenizer_impl)
        converter_graph.node[node_idx].attribute.extend(token_model_attr)

        return converter_graph


class BertTokenizerQADecoder(Step):
    def __init__(self, tokenizer_param: TokenizerParam, name: Optional[str] = None):
        """
        Brief:
            Decode the input_ids to text
        Args:
            tokenizer_param: some essential info to build a tokenizer.
                you can create a TokenizerParam object like:
                    tokenizer_param = TokenizerParam(vocab=tokenizer.vocab, #vocab is dict or file_path)
            name: Optional name of step. Defaults to 'BertTokenizerQADecoder'
        """
        super().__init__(
            ["start_logits", "end_logits", "input_ids"], ["text"], name)
        self._tokenizer_param = tokenizer_param

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        def build_input_declare():
            inputs = []
            for idx, inp in enumerate(self.input_names):
                input_type_str_x, input_shape_str_x = self._get_input_type_and_shape_strs(graph, idx)
                inputs.append(f"{input_type_str_x}[{input_shape_str_x}] {inp}")
            return ",".join(inputs)

        # A unique name for output shape
        prefix_ = f'step_{self.step_num}'
        output_shape_str = f"_{prefix_}_any_len"
        converter_graph = onnx.parser.parse_graph(
            f"""\
            tokenizer_decoder ({build_input_declare()}) 
                => (string[{output_shape_str}] {self.output_names[0]})
            {{
                i64_em = Constant <value = int64[0] {{}}> ()
                i64_1 = Constant <value = int64[1] {{1}}> ()
                i64_0 = Constant <value = int64[1] {{0}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()

                s_position = ArgMax<axis = -1, keepdims = 0>({self.input_names[0]})
                e_position = ArgMax<axis = -1, keepdims = 0>({self.input_names[1]})
                ee_position = Add(e_position,i64_1)
                u_i64_neg1 = Unsqueeze(i64_neg1, i64_0)
                slice_ids= Slice({self.input_names[2]}, s_position, ee_position, i64_neg1)
                {self.output_names[0]} = com.microsoft.extensions.BertTokenizerDecoder (slice_ids, i64_em)
            }}
            """
        )

        attrs = _vocab_to_dict(self._tokenizer_param.vocab_or_file)
        token_model_attr = []
        for attr in attrs:
            token_model_attr.append(onnx.helper.make_attribute(attr, attrs[attr]))

        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == "BertTokenizerDecoder")
        converter_graph.node[node_idx].attribute.extend(token_model_attr)

        return converter_graph
