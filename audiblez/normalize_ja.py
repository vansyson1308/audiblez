# -*- coding: utf-8 -*-
"""
Japanese reading fixes for speech synthesis.

Qwen3-TTS misreads Japanese numbers with counters (3本, 4人, 14日, 1ヶ月...) and some fixed words, while it reads
ordinary kanji text with a natural pitch accent. So only the risky spans are replaced by their hiragana reading
(computed with pyopenjtalk when installed); the rest of the sentence is left as written.
"""
import re

_COUNTERS = sorted([
    '年', '月', '日', '時', '分', '秒', '人', '本', '枚', '個', '匹', '頭', '羽', '回', '度', '歳', '才', '円', '階', '冊', '台', '杯',
    '番', '位', '名', '話', '章', '巻', '件', '通', '着', '足', '軒', 'つ', '週間', '時間', '日間', '年間', '分間', '秒間', 'ヶ月', 'か月',
    'カ月', '箇月', 'ヵ月', '箇所', 'ヶ所', 'か所', 'カ所', '年生', '年前', '年後', '日目', '回目', '人目', '番目', '度目', '代', '世紀',
    '倍', '割', '点', '部', '号', '曲', '泊', '晩', '缶', '粒', '発', '隻', '機', '基', '名様', '人前', '%', '％', 'パーセント',
    'キロ', 'メートル', 'センチ', 'ミリ', 'グラム', 'リットル', 'ドル', 'ユーロ', '万', '億', '兆', '千', '百',
], key=len, reverse=True)
_NUMBER_SPAN = re.compile(r'(?:第)?(?:\d+(?:[.,]\d+)?(?:' + '|'.join(map(re.escape, _COUNTERS)) + r')?)+')  # 第1話

# Irregular readings written in kanji numerals, and heading words that dictionaries misread (間章 -> a name)
_FIXED_READINGS = {
    '二十日': 'はつか', '二十歳': 'はたち', '一人': 'ひとり', '二人': 'ふたり',
}
_HEADING_READINGS = {
    '間章': 'かんしょう', '幕間': 'まくあい', '閑話': 'かんわ', '序章': 'じょしょう', '終章': 'しゅうしょう',
    '番外編': 'ばんがいへん', '後日談': 'ごじつだん', '外伝': 'がいでん',
}
_VOWEL_OF = {}
for vowel, row in (('あ', 'あかさたなはまやらわがざだばぱぁゃ'), ('い', 'いきしちにひみりぎじぢびぴぃ'),
                   ('う', 'うくすつぬふむゆるぐずづぶぷぅゅ'), ('い', 'えけせてねへめれげぜでべぺぇ'),
                   ('う', 'おこそとのほもよろをごぞどぼぽぉょ')):
    for ch in row:
        _VOWEL_OF[ch] = vowel

_G2P = None


def _g2p():
    global _G2P
    if _G2P is None:
        try:
            import pyopenjtalk
            _G2P = pyopenjtalk
        except Exception:
            _G2P = False
    return _G2P or None


def katakana_to_hiragana(text, keep_suffix=0):
    """'ジューイチガツ' -> 'じゅういちがつ' (long vowel marks written out). The last `keep_suffix` characters stay in
    katakana (loanword counters: 'ゴジュッパーセント' -> 'ごじゅっパーセント')."""
    suffix = text[len(text) - keep_suffix:] if keep_suffix else ''
    text = text[:len(text) - keep_suffix] if keep_suffix else text
    out = []
    for ch in text:
        if 'ァ' <= ch <= 'ヶ':
            ch = chr(ord(ch) - 0x60)
        if ch == 'ー' and out:
            ch = _VOWEL_OF.get(out[-1], 'ー')
        out.append(ch)
    return ''.join(out) + suffix


def reading(text):
    """Katakana reading of a Japanese text with pyopenjtalk (None when it is not installed)."""
    g2p = _g2p()
    if g2p is None:
        return None
    try:
        return g2p.g2p(text, kana=True)
    except Exception:
        return None


def normalize_numbers_ja(text):
    """'3本のペン' -> 'さんぼんのペン', '2026年4月14日' -> 'にせんにじゅうろくねんしがつじゅうよっか'."""
    for word, kana in _FIXED_READINGS.items():
        text = text.replace(word, kana)
    if _g2p() is None or not re.search(r'\d', text):
        return text

    def repl(m):
        span = m.group(0)
        kana = reading(span)
        if not kana:
            return span
        loan = re.search(r'(?:%|％|[ァ-ヶー]+)$', span)  # katakana counter (キロ, パーセント...): keep it in katakana
        keep = 0
        if loan:
            keep = len('パーセント') if loan.group(0) in ('%', '％') else len(loan.group(0))
            keep = min(keep, len(kana))
        return katakana_to_hiragana(kana, keep_suffix=keep)
    return _NUMBER_SPAN.sub(repl, text)


def normalize_heading_words_ja(text):
    for word, kana in _HEADING_READINGS.items():
        text = text.replace(word, kana)
    return text
