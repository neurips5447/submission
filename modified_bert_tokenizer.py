from  transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, XLNetTokenizer
import regex as re

class ModifiedBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
    
    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in text.split():
                split_tokens.append(token)
            #for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # If the token is part of the never_split set
                #if token in self.basic_tokenizer.never_split:
                #    split_tokens.append(token)
                #else:
                #    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

class ModifiedRobertaTokenizer(RobertaTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    def _tokenize(self, text):
        # split_tokens = []
        # for token in text.split():
        #     split_tokens.append(token)
        # #for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
        #     # If the token is part of the never_split set
        #     #if token in self.basic_tokenizer.never_split:
        #     #    split_tokens.append(token)
        #     #else:
        #     #    split_tokens += self.wordpiece_tokenizer.tokenize(token)

        # return split_tokens

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend( [(self.bpe(token).split(" "))[0]] )
            #bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


class ModifiedXLNetTokenizer(XLNetTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kwargs
    ):
        super().__init__(
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kwargs)
       
    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        from transformers.file_utils import SPIECE_UNDERLINE

        text = self.preprocess_text(text)

        new_pieces = []

        for word in text.split(" "):
            if not sample:
                pieces = self.sp_model.EncodeAsPieces(word.lower())
            else:
                pieces = self.sp_model.SampleEncodeAsPieces(word, 64, 0.1)
            new_pieces.append(pieces[0])

        return new_pieces

        # new_pieces = []
        # for piece in pieces:
        #     if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
        #         cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
        #         if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        #             if len(cur_pieces[0]) == 1:
        #                 cur_pieces = cur_pieces[1:]
        #             else:
        #                 cur_pieces[0] = cur_pieces[0][1:]
        #         cur_pieces.append(piece[-1])
        #         new_pieces.extend(cur_pieces)
        #     else:
        #         new_pieces.append(piece)

        # return new_pieces
