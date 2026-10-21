"""Bilingual pronunciation frontend used by the reference-conditioned acoustic model."""

import os
import re
import numpy as np
from pypinyin import pinyin, Style
from text import text_to_sequence
from text.cleaners import english_cleaners


def load_g2p():
    import nltk

    resources = [
        "corpora/cmudict",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError as error:
            raise ValueError(
                "Out-of-dictionary English needs NLTK data. Run: python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng"
            ) from error
    from g2p_en import G2p

    return G2p()


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def is_chinese_char(char):
    # including punctuation
    patten = r"[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]"
    if char is None or len(char) == 0 or len(char) > 1:
        return False
    if 0x4E00 <= ord(char) <= 0x9FFF or re.search(patten, char):
        return True
    else:
        return False


def preprocess_en_cn(text, preprocess_config):
    text = re.sub(
        r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+$',
        "",
        text,
    )
    lexicon = read_lexicon(
        os.path.join(preprocess_config["path"]["lexicon_path"], "english_cjy.dict")
    )
    lexicon_pinyin = read_lexicon(
        os.path.join(preprocess_config["path"]["lexicon_path"], "pinyin-lexicon-r.dict")
    )
    g2p = None
    phones = []
    words = re.split(
        r'([\^\*&%#@.,\-?!;:"\'()<>[\]{}…=+~\\$|/]|[\s]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\u4e00-\u9fa5])',
        text,
    )
    # 中文 [\u4e00-\u9fa5]
    # 中文标点 [\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]
    # print("\nwords : ", words)
    for w in words:
        if w is not None:
            if is_chinese_char(w):
                p_pinyin = pinyin(w, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
                p_pinyin = p_pinyin[0]
                p_pinyin = p_pinyin[0]
                if p_pinyin in lexicon_pinyin:
                    phones += lexicon_pinyin[p_pinyin]
                else:
                    phones.append("sp")
            else:
                tmp_cleaned = re.split(r"[\-_\s]", english_cleaners(w))
                # print("tmp_cleaned: ", tem_cleaned)
                for tmp in tmp_cleaned:
                    if not tmp:
                        continue
                    if tmp.lower() in lexicon:
                        phones += lexicon[tmp.lower()]
                    else:
                        if g2p is None:
                            g2p = load_g2p()
                        phones += list(filter(lambda p: p != " ", g2p(tmp.lower())))

    # print("\nwords : ", phones)

    phones = "{" + "}{".join(phones) + "}"
    # print("phones before: ", phones)
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    # print("phones before: ", phones)
    # for p in phones:

    phones = phones.replace("}{", " ")
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, []))
    # print("sequence: ", sequence)
    return np.array(sequence)
