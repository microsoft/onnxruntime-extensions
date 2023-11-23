Module pre_post_processing.steps.nlp
====================================

Classes
-------

`BertTokenizer(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, need_token_type_ids_output: bool = False, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
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

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`BertTokenizerQADecoder(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief:
        Decode the input_ids to text
    Args:
        tokenizer_param: some essential info to build a tokenizer.
            you can create a TokenizerParam object like:
                tokenizer_param = TokenizerParam(vocab=tokenizer.vocab, #vocab is dict or file_path)
        name: Optional name of step. Defaults to 'BertTokenizerQADecoder'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`SentencePieceTokenizer(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, nbest_size=0, alpha=1.0, reverse=False, add_bos=False, add_eos=False, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
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

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`TokenizerParam(vocab_or_file: Union[pathlib.Path, dict], **kwargs)`
: