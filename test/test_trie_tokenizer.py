# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################
import os
import tempfile
import requests

import numpy as np
from unittest import TestCase, main as unittest_main
from onnxruntime_extensions import OrtPyFunction, util, ONNXRuntimeException


# to avoid to install rwkv LM package, we copy the tokenizer code here.
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while (fr != None):
            if (fr.ch != None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if (idx == len(key)):
            if (val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if (self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while (u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if (u.values):
                ret = idx, u, u.values
            if (idx == len(key)):
                break
            ch = key[idx]
        return ret


class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []  # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while (idx < len(src)):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert (idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd'  # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


########################################################################################################


class TestTrieTokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        url = "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt"
        # Create a temporary directory and file path
        temp_dir = tempfile.mkdtemp()
        file_name = os.path.basename(url)  # Gets the file name from the URL
        cls.vocab_file = os.path.join(temp_dir, file_name)
        response = requests.get(url)
        with open(cls.vocab_file, "wb") as f:
            f.write(response.content)

    def test_trie_tokenizer(self):
        tokr = TRIE_TOKENIZER(self.vocab_file)
        src = "I love you"
        tokens = tokr.encode(src)
        self.assertEqual(tokens, [74, 31337, 22799])
        self.assertEqual(tokr.decode(tokens), src)

    def test_ort_trie_tokenizer(self):
        vocab_data = util.read_file(self.vocab_file, 'rb')
        tokr = OrtPyFunction.from_customop("TrieTokenizer", vocab=vocab_data, cpu_only=True)
        tokens = tokr(["I love you"])
        self.assertEqual(list(tokens[0]), [74, 31337, 22799])

        detok = OrtPyFunction.from_customop("TrieDetokenizer", vocab=vocab_data, cpu_only=True)
        self.assertEqual(list(detok(tokens)), ["I love you"])

    def test_invalid_utf8(self):
        vocab_data = util.read_file(self.vocab_file, 'rb')
        detok = OrtPyFunction.from_customop("TrieDetokenizer", vocab=vocab_data, cpu_only=True)
        self.assertRaises(ONNXRuntimeException, detok, np.array([[148]], np.int64))

    def test_parity(self):
        test_sentences = [
            "I am a girl",
            "我是个女孩",
            "私は女の子です",
            "广东人爱吃云吞面，还有腌面、竹升面，车仔面、油渣面、普宁面线、伊面等各种圆扁粗细，加碱水，不加碱水的面",
            "我是个人类",
            "I am a human",
            "that dog is so cute",
            "私はねこむすめです、にゃん♪",
            "宇宙级特大事件！号外号外！"
        ]

        tokr = TRIE_TOKENIZER(self.vocab_file)

        ortx_tokr = OrtPyFunction.from_customop("TrieTokenizer",
                                                vocab=util.read_file(self.vocab_file, 'rb'),
                                                cpu_only=True)
        for s in test_sentences:
            self.assertEqual(tokr.encode(s), list(ortx_tokr([s])[0]))


if __name__ == "__main__":
    unittest_main()
