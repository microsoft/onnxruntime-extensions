# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from collections import OrderedDict

from typing import List, Optional, Tuple, Union
from ..step import Step


class SentencePieceTokenizer(Step):
    def __init__(self, tokenizer, name: Optional[str] = None):
        """
        Args:
            tokenizer: SentencePiece tokenizer from huggingface,
                usually, we can get it by "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
            name: Optional name of step. Defaults to 'SentencePieceTokenizer'

        """
        super().__init__(
            ["inputs", "nbest_size", "alpha", "add_bos", "add_eos", "reverse"], ["input_ids", "attention_mask"], name
        )
        self.tokenizer = tokenizer

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_names = [inp.name for inp in graph.output]
        assert self.input_names[0] in graph_input_names

        # input text
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)

        input_type_str1, input_shape_str1 = (
            self._get_input_type_and_shape_strs(graph, 1)
            if self.input_names[1] in graph_input_names
            else ("int64", "0")
        )
        input_type_str2, input_shape_str2 = (
            self._get_input_type_and_shape_strs(graph, 2)
            if self.input_names[2] in graph_input_names
            else ("float", "0")
        )
        input_type_str3, input_shape_str3 = (
            self._get_input_type_and_shape_strs(graph, 3) if self.input_names[3] in graph_input_names else ("bool", "0")
        )
        input_type_str4, input_shape_str4 = (
            self._get_input_type_and_shape_strs(graph, 4) if self.input_names[4] in graph_input_names else ("bool", "0")
        )
        input_type_str5, input_shape_str5 = (
            self._get_input_type_and_shape_strs(graph, 5) if self.input_names[5] in graph_input_names else ("bool", "0")
        )

        output_shape_str = "batch_size,max_seq_len"
        assert input_type_str0 == "string" and input_type_str1 == "int64" and input_type_str2 == "float"

        assert (len(input_shape_str0.split(",")) == 1) and len(input_shape_str1) == 1 and len(input_shape_str2) == 1

        def build_input_declare():
            input_base = [f"{input_type_str0}[{input_shape_str0}] {self.input_names[0]}"]
            if self.input_names[1] in graph_input_names:
                input_base.append(f"{input_type_str1}[{input_shape_str1}] {self.input_names[1]}")

            if self.input_names[2] in graph_input_names:
                input_base.append(f"{input_type_str2}[{input_shape_str2}] {self.input_names[2]}")

            if self.input_names[3] in graph_input_names:
                input_base.append(f"{input_type_str3}[{input_shape_str3}] {self.input_names[3]}")

            if self.input_names[4] in graph_input_names:
                input_base.append(f"{input_type_str4}[{input_shape_str4}] {self.input_names[4]}")

            if self.input_names[5] in graph_input_names:
                input_base.append(f"{input_type_str5}[{input_shape_str5}] {self.input_names[5]}")

            return ",".join(input_base)

        def build_call_para():
            para_base = [f"{self.input_names[0]}"]
            if self.input_names[1] in graph_input_names:
                para_base.append(f"{self.input_names[1]}")
            else:
                para_base.append(f"i64_0")

            if self.input_names[2] in graph_input_names:
                para_base.append(f"{self.input_names[2]}")
            else:
                para_base.append(f"f32_0")

            if self.input_names[3] in graph_input_names:
                para_base.append(f"{self.input_names[3]}")
            else:
                para_base.append(f"bool_0")

            if self.input_names[4] in graph_input_names:
                para_base.append(f"{self.input_names[4]}")
            else:
                para_base.append(f"bool_1")

            if self.input_names[5] in graph_input_names:
                para_base.append(f"{self.input_names[5]}")
            else:
                para_base.append(f"bool_0")

            return ",".join(para_base)

        def build_forward_declare():
            declare_base = [
                f"i64_0 = Constant <value = int64[1] {{0}}> ()",
                f"f32_0 = Constant <value = float[1] {{0.0}}> ()",
                f"bool_0 = Constant <value = bool[1] {{0}}> ()",
                f"bool_1 = Constant <value = bool[1] {{1}}> ()",
            ]

            declare = [declare_base[0]]
            if self.input_names[2] not in graph_input_names:
                declare.append(declare_base[1])

            if (
                self.input_names[3] not in graph_input_names
                or self.input_names[4] not in graph_input_names
                or self.input_names[5] not in graph_input_names
            ):
                declare.append(declare_base[2])
                declare.append(declare_base[3])

            return "\n".join(declare)

        # Tokenizer does not ideally match with the sentencePiece Model file, so we need to cat the bos_token_id to token. bos_id in model is 1 while 0 in tokenizer
        converter_graph = onnx.parser.parse_graph(
            f"""\
            tokenizer ({build_input_declare()}) 
                => (int64[{output_shape_str}] {self.output_names[0]},int64[{output_shape_str}] {self.output_names[1]})  
            {{
                {build_forward_declare()}
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()
                token,idx = ai.onnx.contrib.SentencepieceTokenizer ({build_call_para()})
                k_start = Constant <value = int32[1] {{{self.tokenizer.bos_token_id}}}> ()
                input_ids_concat02 = Concat <axis = 0> (k_start, token)
                input_ids_bdim = Unsqueeze(input_ids_concat02, i64_0)
                {self.output_names[0]} = Cast <to = 7> (input_ids_bdim)
                attention_mask_i32=Greater({self.output_names[0]}, i64_neg1)
                {self.output_names[1]} = Cast <to = 7> (attention_mask_i32)
            }}
            """
        )

        with open(self.tokenizer.vocab_file, "rb") as f:
            content = f.read()

        token_model_attr = onnx.helper.make_attribute("model", content)
        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == "SentencepieceTokenizer")
        converter_graph.node[node_idx].attribute.append(token_model_attr)

        return converter_graph


class BertTokenizer(Step):
    def __init__(self, tokenizer, name: Optional[str] = None):
        """
        Brief: This step is used to convert the input text into the input_ids, attention_mask, token_type_ids
            support BertTokenizer and HfBertTokenizer, the latter is used for Qa TASK
        Args:
            tokenizer: SentencePiece tokenizer from huggingface,
                usually, we can get it by "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
            name: Optional name of step. Defaults to 'BertTokenizer'

        """
        super().__init__(["inputs"], ["input_ids", "attention_mask", "token_type_ids"], name)
        self.tokenizer = tokenizer

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_names = [inp.name for inp in graph.output]
        for inp in self.input_names:
            assert inp in graph_input_names

        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        output_shape_str = "batch_size,max_seq_len"
        assert input_type_str0 == "string"

        assert len(input_shape_str0.split(",")) in [1, 2], "support either one (classification) or two sentences (QA)"
        onnx_tokenizer_impl = "HfBertTokenizer" if len(input_shape_str0.split(",")) == 2 else "BertTokenizer"

        def build_output_declare():
            output_base = []
            for out in self.output_names:
                output_base.append(f"int64[{output_shape_str}] {out}")

            return ",".join(output_base)

        def get_tokenizer_ret():
            if onnx_tokenizer_impl == "HfBertTokenizer":
                return ",".join(self.output_names)

            return f"ids,types,mask"

        def build_output_imp():
            if onnx_tokenizer_impl == "HfBertTokenizer":
                return ""

            ret_vars = get_tokenizer_ret().split(",")
            ret_vars[1], ret_vars[2] = ret_vars[2], ret_vars[1]
            output_str = [f"i64_0 = Constant <value = int64[1] {{0}}> ()"]

            for i in range(0, len(self.output_names)):
                output_str.append(f"{self.output_names[i]} = Unsqueeze({ret_vars[i]}, i64_0)")

            return "\n".join(output_str)

        def build_input_declare():
            inputs = f"{input_type_str0}[{input_shape_str0}] {self.input_names[0]}"
            return inputs

        def build_tokenizer_call_arg():
            call_args = f"{self.input_names[0]}"
            return call_args

        converter_graph = onnx.parser.parse_graph(
            f"""\
            tokenizer ({build_input_declare()}) 
                => ({build_output_declare()})
            {{
                {get_tokenizer_ret()} = ai.onnx.contrib.{onnx_tokenizer_impl} ({build_tokenizer_call_arg()})
                {build_output_imp()}
            }}
            """
        )

        token_model_attr = []
        bert_tokenizer = self.tokenizer
        ordered_vocab = OrderedDict(sorted(bert_tokenizer.vocab.items(), key=lambda item: int(item[1])))
        vocab = "\n".join(ordered_vocab.keys())
        attrs = dict(vocab_file=vocab)

        attrs["strip_accents"] = (
            1
            if "strip_accents" in bert_tokenizer.init_kwargs and bert_tokenizer.init_kwargs.get("strip_accents")
            else 0
        )
        attrs["do_lower_case"] = 1 if hasattr(bert_tokenizer, "do_lower_case") and bert_tokenizer.do_lower_case else 0

        for attr in attrs:
            attr_value = onnx.helper.make_attribute(attr, attrs[attr])
            token_model_attr.append(attr_value)

        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == onnx_tokenizer_impl)
        converter_graph.node[node_idx].attribute.extend(token_model_attr)

        return converter_graph


class BertTokenizerQATask(Step):
    def __init__(self, name: Optional[str] = None):
        """
        Brief:
            Just copy input_ids for decoder
        Args:
            name: Optional name of step. Defaults to 'BertTokenizerQATask'

        """
        super().__init__(
            ["input_ids", "attention_mask", "token_type_ids"],
            ["input_ids", "attention_mask", "token_type_ids", "input_ids_1"],
            name,
        )

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_names = [inp.name for inp in graph.output]
        for inp in self.input_names:
            assert inp in graph_input_names

        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        output_shape_str = "batch_size,max_seq_len"

        def build_output_declare():
            output_base = []
            for out in self.output_names:
                # used for MaskLM task
                if "predicted_token_id" in out:
                    shape_str = "1"
                else:
                    shape_str = input_shape_str0
                output_base.append(f"{input_type_str0}[{shape_str}] {out}")

            return ",".join(output_base)

        def build_output_imp():
            output_str = []
            for i in range(0, len(self.output_names) - 1):
                output_str.append(f"{self.output_names[i]} = Identity({self.input_names[i]})")

            output_str.append(f"{self.output_names[3]} = Identity({self.input_names[0]})")
            output_str = "\n".join(output_str)

            if len(self.output_names) == 5:
                output_str += f"""\
                mask_id = Constant <value = int64[1] {{{bert_tokenizer.mask_token_id}}}> ()
                zero_or_one = Equal(dim_input_ids, mask_id)
                zero_or_one_c = Cast <to = 2> (zero_or_one)
                {self.output_names[4]} = ArgMax<axis = -1>(zero_or_one_c)
                """

            return output_str

        def build_input_declare():
            inputs = []

            for i in range(0, len(self.input_names)):
                input_type_str_x, input_shape_str_x = self._get_input_type_and_shape_strs(graph, i)
                inputs.append(f"{input_type_str_x}[{input_shape_str_x}] {self.input_names[i]}")

            return ",".join(inputs)

        def build_tokenizer_call_arg():
            call_args = f"{self.input_names[0]}"
            return call_args

        converter_graph = onnx.parser.parse_graph(
            f"""\
            qa_task ({build_input_declare()}) 
                => ({build_output_declare()})
            {{
                {build_output_imp()}
            }}
            """
        )

        return converter_graph


class BertTokenizerQATaskDecoder(Step):
    def __init__(self, tokenizer, name: Optional[str] = None):
        """
        Brief:
            Decode the input_ids to text
        Args:
            tokenizer: SentencePiece tokenizer from huggingface,
                usually, we can get it by "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
            name: Optional name of step. Defaults to 'BertTokenizerQATaskDecoder'

        """
        super().__init__(["start_logits", "end_logits", "input_ids_1"], ["text"], name)
        self.tokenizer = tokenizer

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_names = [inp.name for inp in graph.output]
        for inp in graph_input_names:
            if self.input_names[2] in inp:
                self.input_names[2] = inp
        is_input_names_in_graph = [inp in graph_input_names for inp in self.input_names]
        assert all(is_input_names_in_graph), f"Input names {self.input_names} not in graph inputs {graph_input_names}"

        def build_input_declare():
            inputs = []
            for i in range(0, len(self.input_names)):
                input_type_str_x, input_shape_str_x = self._get_input_type_and_shape_strs(graph, i)
                inputs.append(f"{input_type_str_x}[{input_shape_str_x}] {self.input_names[i]}")
            return ",".join(inputs)

        output_shape_str = "any_len"
        converter_graph = onnx.parser.parse_graph(
            f"""\
            tokenizer_decoder ({build_input_declare()}) 
                => (string[{output_shape_str}] {self.output_names[0]})
            {{
                i64_em = Constant <value = int64[0] {{}}> ()
                i64_1 = Constant <value = int64[1] {{1}}> ()
                i64_0 = Constant <value = int64[1] {{0}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()

                s_position = ArgMax<axis = -1, keepdims = 0>({self.input_names[1]})
                e_position = ArgMax<axis = -1, keepdims = 0>({self.input_names[2]})
                ee_position = Add(e_position,i64_1)
                u_i64_neg1 = Unsqueeze(i64_neg1, i64_0)
                slice_ids= Slice({self.input_names[0]}, s_position, ee_position, i64_neg1)
                {self.output_names[0]} = ai.onnx.contrib.BertTokenizerDecoder (slice_ids, i64_em)
            }}
            """
        )

        bert_tokenizer = self.tokenizer
        ordered_vocab = OrderedDict(sorted(bert_tokenizer.vocab.items(), key=lambda item: int(item[1])))
        vocab = "\n".join(ordered_vocab.keys())
        attrs = dict(vocab_file=vocab)
        token_model_attr = []
        for attr in attrs:
            attr_value = onnx.helper.make_attribute(attr, attrs[attr])
            token_model_attr.append(attr_value)

        node_idx = next(i for i, v in enumerate(converter_graph.node) if v.op_type == "BertTokenizerDecoder")
        converter_graph.node[node_idx].attribute.extend(token_model_attr)
        return converter_graph


class SequenceClassify(Step):
    def __init__(self, name: Optional[str] = None):
        """
        Brief:
            Decode the input_ids to text
        Args:
            tokenizer: SentencePiece tokenizer from huggingface,
                usually, we can get it by "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
            name: Optional name of step. Defaults to 'BertTokenizerQATaskDecoder'

        """
        super().__init__(["logits"], ["index"], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        graph_input_names = [inp.name for inp in graph.output]

        is_input_names_in_graph = [inp in graph_input_names for inp in self.input_names]
        assert all(is_input_names_in_graph), f"Input names {self.input_names} not in graph inputs {graph_input_names}"

        def build_input_declare():
            inputs = []
            for i in range(0, len(self.input_names)):
                input_type_str_x, input_shape_str_x = self._get_input_type_and_shape_strs(graph, i)
                inputs.append(f"{input_type_str_x}[{input_shape_str_x}] {self.input_names[i]}")
            return ",".join(inputs)

        output_shape_str = "batch"
        converter_graph = onnx.parser.parse_graph(
            f"""\
            classify ({build_input_declare()}) 
                => (int64[{output_shape_str}] {self.output_names[0]})
            {{
                {self.output_names[0]} = ArgMax<axis = -1, keepdims=0>({self.input_names[0]})
            }}
            """
        )
        
        return converter_graph