# -*- coding: utf-8 -*-
"""
Korean number normalization for speech synthesis.

Korean has two number systems and the counter decides which one is read: native Korean numbers for counting
things, people, hours, age... (세 명, 두 시, 스무 살) and Sino-Korean numbers for dates, minutes, money,
floors, measures... (이천이십육 년, 삼십 분, 오천 원). Neural TTS models read digits inconsistently, so we spell
them out. Without a counter, the Sino-Korean reading is used.
"""
import re

_SINO_DIGITS = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
_NATIVE_UNITS = {1: '하나', 2: '둘', 3: '셋', 4: '넷', 5: '다섯', 6: '여섯', 7: '일곱', 8: '여덟', 9: '아홉'}
_NATIVE_UNITS_ATTRIBUTIVE = {1: '한', 2: '두', 3: '세', 4: '네'}  # before a counter: 한 명, 두 개, 세 시
_NATIVE_TENS = {1: '열', 2: '스물', 3: '서른', 4: '마흔', 5: '쉰', 6: '예순', 7: '일흔', 8: '여든', 9: '아흔'}

# counters read with native Korean numbers (1-99); longest first
_NATIVE_COUNTERS = sorted([
    '번째', '시간', '사람', '마리', '그릇', '켤레', '송이', '자루', '가지', '군데', '걸음', '바퀴', '개', '명', '살', '시',
    '번', '권', '잔', '병', '대', '채', '벌', '장', '곳', '달', '줄', '통', '쌍', '알', '포기', '덩이', '조각', '방울',
    '모금', '숟가락', '봉지', '상자', '박스', '접시', '척', '그루', '자리', '배', '편',
], key=len, reverse=True)
# counters read with Sino-Korean numbers (also the default)
_SINO_COUNTERS = sorted([
    '개월', '년', '월', '일', '주', '분', '초', '원', '층', '호', '회', '학년', '세', '도', '퍼센트', '킬로그램', '킬로미터',
    '킬로', '미터', '센티미터', '센티', '밀리', '그램', '리터', '쪽', '페이지', '화', '등', '위', '차', '년대', '세기', '인분',
    '달러', '엔', '위안', '유로', '평', '인치', '점', '번지', '동',
], key=len, reverse=True)
_MONTHS = {6: '유월', 10: '시월'}  # 6월 and 10월 drop a final consonant


def sino(n: int) -> str:
    """Sino-Korean reading: 2026 -> 이천이십육, 15000 -> 만 오천, 123456789 -> 일억 이천삼백사십오만 육천칠백팔십구."""
    if n == 0:
        return '영'
    if n < 0:
        return '마이너스 ' + sino(-n)
    groups = []
    for unit in ['', '만', '억', '조', '경']:
        n, part = divmod(n, 10000)
        groups.append((part, unit))
        if n == 0:
            break
    words = []
    for part, unit in reversed(groups):
        if part == 0:
            continue
        w = _sino_below_10000(part)
        if unit == '만' and part == 1:
            w = ''  # 10000 -> 만 (not 일만)
        words.append(w + unit)
    return ' '.join(words)


def _sino_below_10000(n: int) -> str:
    out = ''
    for value, name in ((1000, '천'), (100, '백'), (10, '십')):
        d, n = divmod(n, value)
        if d:
            out += ('' if d == 1 else _SINO_DIGITS[d]) + name
    return out + _SINO_DIGITS[n]


def native(n: int, attributive: bool = True) -> str:
    """Native Korean reading for 1-99: 3 -> 세 (before a counter) / 셋, 20 -> 스무 / 스물, 21 -> 스물한."""
    if not 1 <= n <= 99:
        return sino(n)
    tens, units = divmod(n, 10)
    if attributive and n == 20:
        return '스무'
    word = _NATIVE_TENS.get(tens, '')
    if units:
        word += (_NATIVE_UNITS_ATTRIBUTIVE.get(units) if attributive and units in _NATIVE_UNITS_ATTRIBUTIVE
                 else _NATIVE_UNITS[units])
    return word


def _decimal(s: str) -> str:
    whole, _, frac = s.partition('.')
    return sino(int(whole)) + ' 점 ' + ''.join(_SINO_DIGITS[int(d)] or '영' for d in frac)


# longest first across both lists: 2개월 is 이 개월 (Sino-Korean 개월), not 두 개 + 월; 3번지 is 삼 번지
_COUNTER_RE = '|'.join(map(re.escape, sorted(set(_NATIVE_COUNTERS + _SINO_COUNTERS), key=len, reverse=True)))
_NUMBER_RE = re.compile(rf'(?<![\d.,])(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?(\s?)({_COUNTER_RE}|%)?')


def _replace(m: re.Match, heading=False) -> str:
    digits, frac, space, counter = m.group(1).replace(',', ''), m.group(2), m.group(3), m.group(4)
    n = int(digits)
    if len(digits) > 15:  # phone numbers, codes: digit by digit
        return ' '.join(_SINO_DIGITS[int(d)] or '공' for d in digits) + space + (counter or '')
    if frac:
        word = _decimal(digits + frac)
        return word + (' 퍼센트' if counter == '%' else (' ' + counter if counter else ''))
    if counter == '%':
        return sino(n) + ' 퍼센트'
    if counter is None:
        return sino(n) + space
    if counter == '월' and n in _MONTHS:
        return _MONTHS[n]
    if m.string[max(0, m.start() - 1):m.start()] == '제':  # ordinal: 제3장 -> 제삼장, 제1화 -> 제일화 (one word)
        return sino(n) + counter
    if counter in _NATIVE_COUNTERS and 1 <= n <= 99 and not heading:
        return native(n) + ' ' + counter
    return sino(n) + ' ' + counter


def normalize_numbers_ko(text: str, heading: bool = False) -> str:
    """heading=True: in chapter headings ("3장", "12화") numbers are ordinals, read in Sino-Korean."""
    # Times "10:30" -> 열 시 삼십 분
    text = re.sub(r'(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])',
                  lambda m: native(int(m.group(1))) + ' 시' + (f' {sino(int(m.group(2)))} 분' if int(m.group(2)) else ''),
                  text)
    text = re.sub(r'₩\s?(\d[\d,]*)', lambda m: m.group(1) + '원', text)
    text = re.sub(r'\$\s?(\d[\d,.]*)', lambda m: m.group(1) + '달러', text)
    return _NUMBER_RE.sub(lambda m: _replace(m, heading), text)
