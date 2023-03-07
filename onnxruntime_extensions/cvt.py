from ._cuops import CustomOpConverter

class HFTokenizerConverter(CustomOpConverter):
    def __init__(self, processor):
        self.processor = processor

    def bpe_decoder(self, **kwargs):
        decoder = self.processor.tokenizer.decoder
        id_vocab = "\n".join([decoder[_idx] for _idx in sorted(decoder)])
        byte_decoder = self.processor.tokenizer.byte_decoder
        byte_decoder = "\n".join(["{}\t{}".format(
            _c.encode("utf-8"), str(byte_decoder[_c])) for _c in sorted(byte_decoder)])
        kwargs.update({
            "id_vocab": id_vocab,
            "byte_decoder": byte_decoder,
            "added_tokens": "",
            "all_special_ids": "",
            "skip_special_tokens": kwargs.get("skip_special_tokens", False)
            })
        
        return kwargs
