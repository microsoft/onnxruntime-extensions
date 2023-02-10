Module pre_post_processing.steps.nlp
====================================

Classes
-------

`AttributeDict(*args, **kwargs)`
:   dict() -> new empty dictionary
    dict(mapping) -> new dictionary initialized from a mapping object's
        (key, value) pairs
    dict(iterable) -> new dictionary initialized as if via:
        d = {}
        for k, v in iterable:
            d[k] = v
    dict(**kwargs) -> new dictionary initialized with the name=value pairs
        in the keyword argument list.  For example:  dict(one=1, two=2)

    ### Ancestors (in MRO)

    * builtins.dict

`BertTokenizer(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief: This step is used to convert the input text into the input_ids, attention_mask, token_type_ids.
        It support BertTokenizer and HfBertTokenizer, the former can only has one queries, the latter have two
    Args:
        tokenizer_map: some essential infos, If you export tokenizer from hugging-face,
        such as "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
        You may need to provide the following:
            tokenizer_param = TokenizerParam(vocab=tokenizer.vocab, #vocab is dict or file_path,
                                strip_accents = True or False (Optional),
                                do_lower_case = True or False (Optional),
                                )
    
        name: Optional name of step. Defaults to 'BertTokenizer'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`BertTokenizerQATask(name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief:
        Just duplicate input_ids for decoder(TokenizerDecoder). For tasks like 'BertTokenizerQADecoder' which need to use the same input_ids for decoder.
        However, input_ids has its consumers, it will merged and removed in the next step. So we need to duplicate one.
        The new output 'input_ids_1' will be kept as a new output in graph.
    Args:
        name: Optional name of step. Defaults to 'BertTokenizerQATask'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`BertTokenizerQADecoder(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief:
        Decode the input_ids to text
    Args:
        tokenizer_map: some essential infos, If you export tokenizer from hugging-face,
        such as "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
        You may need to provide the following:
            tokenizer_param = TokenizerParam(vocab=tokenizer.vocab, #vocab is dict or file_path)
        name: Optional name of step. Defaults to 'BertTokenizerQADecoder'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`SentencePieceTokenizer(tokenizer_param: pre_post_processing.steps.nlp.TokenizerParam, name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief:
        SentencePieceTokenizer is a bit special here. Most likely, users only want to use it like Bert-Tokenizer which has one "Text" input,
        we support taking the other parameter as optional and use the default value such as the same usage in Hugging-Face.
    
    Args:
        tokenizer_map: some essential infos, If you export tokenizer from hugging-face,
        such as "tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)"
        You may need to provide the following:
            tokenizer_param = TokenizerParam(vocab_size=tokenizer.vocab_size,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id = tokenizer.eos_token_id,)
        name: Optional name of step. Defaults to 'SentencePieceTokenizer'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`SequenceClassify(name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Brief:
        Convert max logit in logits array to index of classes, which used in classify task.
    Args:
        name: Optional name of step. Defaults to 'SequenceClassify'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`TokenizerParam(vocab_or_file: Union[pathlib.Path, dict], **kwargs)`
:   

    ### Methods

    `assigned_with_kwargs(self, **kwargs)`
    :