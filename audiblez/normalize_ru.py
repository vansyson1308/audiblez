# -*- coding: utf-8 -*-
"""
Russian text normalization for speech synthesis.

Neural TTS models read Russian much better when the text is fully spelled out: numbers in the grammatically
correct case and gender, expanded abbreviations, restored "ё" (otherwise "еще" is read "yeshche" instead of
"yeshcho"), no stress marks, and normalized punctuation.

Number agreement uses pymorphy3 (morphological analysis of the following noun) when available, and falls back
to preposition-based heuristics otherwise.
"""
import re
from functools import lru_cache

from num2words import num2words

# ---------------------------------------------------------------------------------------------------------------------
# Morphology helpers
# ---------------------------------------------------------------------------------------------------------------------

_MORPH = None


def _morph():
    global _MORPH
    if _MORPH is None:
        try:
            import pymorphy3
            _MORPH = pymorphy3.MorphAnalyzer()
        except Exception:  # pragma: no cover - optional dependency
            _MORPH = False
    return _MORPH or None


_CASE_MAP = {'nomn': 'n', 'gent': 'g', 'gen1': 'g', 'gen2': 'g', 'datv': 'd', 'accs': 'a', 'acc2': 'a',
             'ablt': 'i', 'loct': 'p', 'loc1': 'p', 'loc2': 'p', 'voct': 'n'}
_GENDER_MAP = {'masc': 'm', 'femn': 'f', 'neut': 'n', 'ms-f': 'm'}

_PREP_CASE = {}
for _w in ('без', 'для', 'до', 'из', 'из-за', 'из-под', 'от', 'у', 'около', 'после', 'кроме', 'вместо', 'мимо',
           'среди', 'против', 'возле', 'вокруг', 'вдоль', 'с', 'со', 'свыше', 'более', 'менее', 'больше', 'меньше',
           'порядка', 'начиная', 'вне', 'внутри', 'позади', 'посреди', 'ради', 'насчёт', 'насчет', 'близ'):
    _PREP_CASE[_w] = 'g'
for _w in ('к', 'ко', 'согласно', 'благодаря', 'вопреки', 'навстречу'):
    _PREP_CASE[_w] = 'd'
for _w in ('между', 'над', 'надо', 'перед', 'передо'):
    _PREP_CASE[_w] = 'i'
for _w in ('о', 'об', 'обо', 'при'):
    _PREP_CASE[_w] = 'p'


class NounInfo:
    __slots__ = ('case', 'gender', 'plural', 'animate', 'lemma')

    def __init__(self, case, gender, plural, animate, lemma):
        self.case, self.gender, self.plural, self.animate, self.lemma = case, gender, plural, animate, lemma

    def __repr__(self):
        return f'NounInfo({self.lemma}, {self.case}, {self.gender}, plural={self.plural})'


@lru_cache(maxsize=20000)
def _noun_parses(word):
    """All NOUN analyses of a word as NounInfo tuples, most probable first."""
    morph = _morph()
    if morph is None:
        return ()
    out = []
    for p in morph.parse(word):
        tag = p.tag
        if 'NOUN' not in tag:
            continue
        case = _CASE_MAP.get(tag.case)
        if case is None:
            continue
        gender = _GENDER_MAP.get(tag.gender)
        if gender is None:
            try:
                gender = _GENDER_MAP.get(p.normalized.tag.gender)
            except Exception:
                gender = None
        out.append(NounInfo(case, gender or 'm', tag.number == 'plur', 'anim' in tag, p.normal_form))
    return tuple(out)


@lru_cache(maxsize=20000)
def _is_adjective_like(word):
    morph = _morph()
    if morph is None:
        return False
    p = morph.parse(word)[0]
    return any(t in p.tag for t in ('ADJF', 'PRTF', 'NUMR', 'Anum'))


_WORD_RE = re.compile(r'[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁё]+)?')


def _following_noun(text, pos, max_words=3):
    """NOUN analyses of the noun right after position `pos` (skipping adjectives), within the same phrase."""
    i = pos
    for _ in range(max_words):
        i += re.match(r'[ \t«"„“(]*', text[i:]).end()
        wm = _WORD_RE.match(text, i)
        if not wm:
            return ()
        word = wm.group(0).lower()
        if _is_adjective_like(word):
            i = wm.end()
            continue
        return _noun_parses(word)
    return ()


def _preceding_word(text, pos):
    m = re.search(r'([А-Яа-яЁё-]+)[ \t«"(]*$', text[:pos])
    return m.group(1).lower() if m else ''


def _prep_case(text, pos):
    return _PREP_CASE.get(_preceding_word(text, pos))


# ---------------------------------------------------------------------------------------------------------------------
# Number words
# ---------------------------------------------------------------------------------------------------------------------

def cardinal(n, case='n', gender='m', animate=False):
    s = num2words(n, lang='ru', case=case, gender=gender, animate=animate)
    first, _, rest = s.partition(' ')
    if first in ('одна', 'одной', 'одну') and rest.startswith('тысяч'):
        s = rest  # "одна тысяча восемьсот" -> "тысяча восемьсот"
    return s


def ordinal(n, case='n', gender='m', plural=False, animate=False):
    return num2words(n, lang='ru', to='ordinal', case=case, gender=gender, plural=plural, animate=animate)


def _decimal_words(s):
    try:
        return num2words(float(s.replace(',', '.')), lang='ru')
    except Exception:
        return s


_MONTHS_GEN = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября',
               'ноября', 'декабря']
_MONTH_LEMMAS = {'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь',
                 'ноябрь', 'декабрь'}


def number_words(n, text, start, end, suffix=None):
    """Spell out the integer n found at text[start:end], agreeing with the context."""
    prep = _prep_case(text, start)
    nouns = _following_noun(text, end)
    range_second = False
    if not nouns and not suffix:
        # first number of a range "1812 — 1815 годах", "5 — 7 лет": agree with the noun after the second number
        rm = re.match(r'[ \t]*—[ \t]*\d+(?:-[а-яё]{1,3})?', text[end:])
        if rm:
            nouns = _following_noun(text, end + rm.end())
            range_second = True
    elif re.search(r'\d[ \t]*—[ \t]*$', text[:start]):
        range_second = True
    noun = nouns[0] if nouns else None

    if suffix:
        return _ordinal_from_suffix(n, suffix.lower(), nouns, prep)

    # Dates: "7 мая" -> "седьмое мая" (ordinal neuter, case from the preposition)
    if 1 <= n <= 31 and noun is not None and noun.lemma in _MONTH_LEMMAS:
        return ordinal(n, case=prep or _date_case(text, start, end), gender='n')
    # The year of a date written without "года": "14 ноября 1978" -> "... тысяча девятьсот семьдесят восьмого года"
    if 100 <= n <= 2199 and (noun is None or noun.lemma != 'год') and _preceding_word(text, start) in _MONTHS_GEN:
        return ordinal(n, case='g', gender='m') + ' года'
    # "в 1978," with no noun after it: a year -> "в тысяча девятьсот семьдесят восьмом году"
    if (1700 <= n <= 2199 and noun is None and _preceding_word(text, start) in ('в', 'во')
            and re.match(r'[ \t]*(?:[,.;:!?…»)]|—|$)', text[end:])):
        return ordinal(n, case='p', gender='m') + ' году'
    # Centuries and years are ordinals: "XIX век", "в 1812 году", "весной 1834 года", "в 1812 — 1815 годах"
    if noun is not None and ((not noun.plural and (noun.lemma == 'век' or (noun.lemma == 'год' and n >= 100)))
                             or (noun.plural and ((noun.lemma == 'год' and n >= 1000)
                                                  or (noun.lemma == 'век' and range_second)))):
        return ordinal(n, case=noun.case, gender='m')

    last, last2 = n % 10, n % 100
    one = last == 1 and last2 != 11
    few = 2 <= last <= 4 and not 12 <= last2 <= 14
    for nn in nouns:
        if nn.plural and nn.case in ('d', 'i', 'p'):  # "в 5 томах", "с 5 друзьями"
            return cardinal(n, case=nn.case, gender=nn.gender, animate=nn.animate)
        if nn.plural and nn.case == 'g':
            if few or one:  # "до 2 часов"
                return cardinal(n, case='g', gender=nn.gender, animate=nn.animate)
            return cardinal(n, case=prep if prep in ('g', 'd', 'i', 'p') else 'n', gender=nn.gender)  # "5 лет"
        if not nn.plural and one:  # "21 день", "1 книгу"
            return cardinal(n, case=nn.case, gender=nn.gender, animate=nn.animate)
        if not nn.plural and few and nn.case == 'g':
            if prep == 'g':  # "у 3 подъезда" -> "у третьего подъезда"
                return ordinal(n, case='g', gender=nn.gender)
            return cardinal(n, gender=nn.gender)  # "2 книги"
    if noun is not None and not noun.plural and noun.case in ('d', 'i', 'p') and not one:
        return ordinal(n, case=noun.case, gender=noun.gender)  # "в 17 нумере", "на 5 странице"
    return cardinal(n, case=prep or 'n', gender=noun.gender if noun is not None else 'm')


_NOMINATIVE_DATE_PREV = {'было', 'был', 'будет', 'сегодня', 'завтра', 'вчера', 'наступило', 'настало', 'наступает',
                         'это', 'стояло', 'шло', 'число'}


def _date_case(text, start, end):
    """'Седьмого мая он вернулся' (adverbial, genitive) vs 'Было седьмое мая.' / letter dates (nominative)."""
    if _preceding_word(text, start) in _NOMINATIVE_DATE_PREV:
        return 'n'
    # Standalone date (only the month and maybe a year until the end of the sentence) -> nominative
    rest = re.split(r'[.!?;:](?!\d)', text[end:], maxsplit=1)[0]
    words = re.findall(r'[А-Яа-яЁё]+|\d+', rest)
    year_words = re.compile(r'^(?:\d+|год[а-я]*|г|тысяч[а-я]*|\w*сот[а-я]*|сто|двест[а-я]*|\w*дцат[а-я]*|'
                            r'\w*десят[а-я]*|сорок[а-я]*|девяност[а-я]*|перв[а-я]*|втор[а-я]*|трет[а-я]*|'
                            r'четв[её]рт[а-я]*|пят[а-я]*|шест[а-я]*|седьм[а-я]*|восьм[а-я]*|девят[а-я]*|'
                            r'один|одна|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)$', re.I)
    return 'n' if all(year_words.match(w) for w in words[1:]) else 'g'


# suffix -> candidate (case, gender, plural) forms, most likely first
_SUFFIX_FORMS = {
    'й': [('n', 'm', False), ('p', 'f', False), ('g', 'f', False), ('d', 'f', False), ('i', 'f', False)],
    'я': [('n', 'f', False)],
    'ю': [('a', 'f', False)],
    'е': [('n', 'n', False), ('a', 'n', False), ('n', 'm', True), ('a', 'm', True)],
    'го': [('g', 'm', False), ('g', 'n', False), ('a', 'm', False)],
    'му': [('d', 'm', False), ('d', 'n', False)],
    'м': [('p', 'm', False), ('p', 'n', False), ('i', 'm', False), ('i', 'n', False), ('d', 'm', True)],
    'х': [('p', 'm', True), ('g', 'm', True), ('a', 'm', True)],
    'ми': [('i', 'm', True)],
}
_SUFFIX_ALIASES = {'ый': 'й', 'ий': 'й', 'ой': 'й', 'ая': 'я', 'яя': 'я', 'ую': 'ю', 'юю': 'ю', 'ое': 'е',
                   'ее': 'е', 'ые': 'е', 'ие': 'е', 'ого': 'го', 'его': 'го', 'ому': 'му', 'ему': 'му', 'ом': 'м',
                   'ем': 'м', 'ым': 'м', 'им': 'м', 'ых': 'х', 'их': 'х', 'ыми': 'ми', 'ими': 'ми'}


def _ordinal_from_suffix(n, suffix, nouns, prep):
    forms = _SUFFIX_FORMS.get(_SUFFIX_ALIASES.get(suffix, suffix))
    if not forms:
        return cardinal(n)
    for nn in nouns:
        for case, gender, plural in forms:
            if case == nn.case and plural == nn.plural and (plural or gender == nn.gender):
                return ordinal(n, case=case, gender=gender, plural=plural)
    if prep:
        for case, gender, plural in forms:
            if case == prep:
                return ordinal(n, case=case, gender=gender, plural=plural)
    case, gender, plural = forms[0]
    return ordinal(n, case=case, gender=gender, plural=plural)


_COLLOQUIAL_CARDINAL_SUFFIXES = {'ти', 'ми', 'ью', 'ух', 'ёх', 'ех', 'ри', 'ре', 'мя'}


def _is_colloquial_cardinal(n, suffix):
    """'2-х', '3-х', '5-ти', '7-ми' are cardinals ('двух', 'пяти'), while '1990-х' or '5-м' are ordinals."""
    if suffix in _COLLOQUIAL_CARDINAL_SUFFIXES and not (suffix == 'ми' and n % 10 not in (7, 8)):
        return True
    return suffix == 'х' and n < 100 and 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14


def _colloquial_cardinal(n, suffix, text, start, end):
    nouns = _following_noun(text, end)
    noun = nouns[0] if nouns else None
    if suffix == 'ью':
        case = 'i'
    elif noun is not None and noun.plural and noun.case in ('g', 'd', 'i', 'p'):
        case = noun.case
    else:
        case = _prep_case(text, start) or 'g'
    return cardinal(n, case=case, gender=noun.gender if noun is not None else 'm')


_ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
# Roman numerals, also accepting the Cyrillic look-alikes Х and І that are frequently typed instead of X and I
_ROMAN_TOKEN = r'(?=[MDCLXVIХІ])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?[XХ]{0,3})(?:[IІ][XХ]|[IІ]V|V?[IІ]{0,3})'
_LOOKALIKE = str.maketrans({'Х': 'X', 'І': 'I'})


def roman_to_int(s):
    s = (s or '').translate(_LOOKALIKE).upper()
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total or None


# ---------------------------------------------------------------------------------------------------------------------
# ё restoration (only words whose "е" spelling cannot be another word)
# ---------------------------------------------------------------------------------------------------------------------

_HARD_ADJ = ['ый', 'ая', 'ое', 'ые', 'ого', 'ой', 'ому', 'ым', 'ом', 'ую', 'ых', 'ыми']
_SOFT_ADJ = ['ий', 'ая', 'ое', 'ие', 'ого', 'ой', 'ому', 'им', 'ом', 'ую', 'их', 'ими']
_NOUN_HARD_M = ['', 'а', 'у', 'ом', 'е', 'ы', 'ов', 'ам', 'ами', 'ах']
_NOUN_FEM_A = ['а', 'ы', 'е', 'у', 'ой', '', 'ам', 'ами', 'ах']
_NOUN_FEM_SH = ['а', 'и', 'е', 'у', 'ей', '', 'ам', 'ами', 'ах']
_NOUN_FEM_YA = ['я', 'и', 'е', 'ю', 'ей', 'ь', 'ям', 'ями', 'ях']
_NOUN_FEM_KA = ['а', 'и', 'е', 'у', 'ой', 'ам', 'ами', 'ах']
_NOUN_ONOK = ['ок', 'ка', 'ку', 'ком', 'ке']
_NOUN_OK = ['ок', 'ка', 'ку', 'ком', 'ке', 'ки', 'ков', 'кам', 'ками', 'ках']
_NOUN_SOFT_M = ['ь', 'я', 'ю', 'ем', 'е', 'и', 'ей', 'ям', 'ями', 'ях']

_YO_ADJ_HARD_STEMS = (
    'чёрн жёлт зелён тёмн тёпл тяжёл весёл мёртв твёрд серьёзн учён солён копчён печён варён кручён мочён '
    'толчён жжён смышлён надёжн безнадёжн пёстр отчётлив расчётлив звёздн берёзов шёлков ёлочн самолётн '
    'утончённ изощрённ смущённ огорчённ удивлённ оживлённ раздражённ освещённ обречённ увлечённ истощённ '
    'посвящённ заключённ приближённ сражённ поражённ напряжённ вооружённ обнажённ удалённ отдалённ утомлённ '
    'влюблённ оскорблённ углублённ упрощённ запрещённ защищённ возмущённ восхищённ решённ лишённ окружённ '
    'поглощённ погружённ ожесточённ пленённ сплочённ умудрённ разгорячённ облегчённ клеймённ зачёркнут '
    'перечёркнут подчёркнут трёхэтажн четвёрт'
).split()
_YO_ADJ_SOFT_STEMS = 'лёгк жёстк чётк'.split()
_YO_NOUN_M_STEMS = ('самолёт полёт вертолёт пулемёт налёт взлёт переплёт отчёт расчёт учёт приём подъём объём '
                    'актёр режиссёр шофёр дирижёр партнёр монтёр боксёр гримёр стажёр суфлёр ухажёр клён шёпот '
                    'жёрнов Фёдор Семён Артём Парфён').split()
_YO_NOUN_OK_STEMS = 'посёл свёрт'.split()  # посёлок, свёрток
_YO_NOUN_SOFT_M_STEMS = 'жёлуд'.split()
_YO_NOUN_FEM_A_STEMS = 'берёз Алён Матрён Фёкл свёкл пощёчин тёзк'.split()
_YO_NOUN_FEM_SH_STEMS = 'Алёш Лёш тёщ'.split()
_YO_NOUN_FEM_YA_STEMS = 'тёт'.split()
_YO_NOUN_FEM_KA_STEMS = 'ёлк тётк плёнк пчёлк чёлк печёнк сгущёнк тёрк щётк четвёрк тётушк чёрточк'.split()
_YO_ONOK_STEMS = ('ребён котён телён утён цыплён поросён жеребён гусён лисён слонён львён тигрён орлён бесён '
                  'чертён опён ягнён козлён оленён лосён').split()
_YO_WORDS = (
    'её неё моё твоё своё чьё ничьё твоём своём ещё всё-таки причём нипочём почём вперёд вдвоём втроём '
    'живьём заём наём лёд мёд мёда мёду мёдом пёс лён шёлк шёлком ёж ёжик ёжика ёжику ёжиком ёжике ёжики '
    'огонёк денёк пенёк паренёк мотылёк уголёк конёк кулёк стебелёк ручеёк зверёк счёту счётом счёте Пётр '
    'твёрдо серьёзно надёжно безнадёжно отчётливо расчётлив мёртв тяжёл смущён удивлён решён поражён влюблён '
    'лишён окружён огорчён раздражён возмущён восхищён оживлён утомлён увлечён обречён заключён запрещён '
    'защищён погружён поглощён напряжён вооружён обнажён твёрдость твёрдости твёрдостью лёгкость лёгкости '
    'лёгкостью серьёзность серьёзности серьёзностью чёт нечёт трёх четырёх трём четырём трёхсот четырёхсот '
    'трёхстах свёкор свёкра свёкру свёкром лётчик лётчика лётчику лётчиком лётчике лётчики лётчиков '
    'лётчикам лётчиками лётчиках чёртов чёртова чёртово чёртовы чёртовой чёртову чёртовым чёртовых чёртовом '
    'шёл пришёл ушёл нашёл пошёл вошёл зашёл подошёл прошёл перешёл отошёл обошёл сошёл дошёл взошёл '
    'произошёл нашёлся вёл привёл увёл завёл отвёл подвёл провёл перевёл развёл навёл произвёл изобрёл '
    'приобрёл обрёл нёс принёс унёс занёс отнёс поднёс пронёс перенёс вёз привёз увёз отвёз завёз повёз грёб '
    'плёл цвёл расцвёл отцвёл лёг слёг прилёг залёг полёг жёг сжёг поджёг зажёг обжёг стёр растёр простёр '
    'стёрся упёрся замёрз мёрз щёлкнул щёлкнула щёлкнули щёлкнуть щёлкать щёлкает '
    'идёт идёшь идём идёте пойдёт пойдёшь пойдём пойдёте придёт придёшь придём придёте найдёт найдёшь найдём '
    'найдёте уйдёт уйдёшь уйдём уйдёте войдёт войдёшь войдём войдёте подойдёт подойдёшь подойдём подойдёте '
    'пройдёт пройдёшь пройдём пройдёте дойдёт дойдёшь дойдём дойдёте зайдёт зайдёшь зайдём зайдёте перейдёт '
    'перейдёшь перейдём обойдёт обойдёшь обойдём сойдёт сойдёшь сойдём отойдёт отойдёшь отойдём произойдёт '
    'найдётся придётся пройдётся обойдётся живёт живёшь живём живёте даёт даёшь даём даёте встаёт встаёшь '
    'встаём встаёте поёт поёшь поёте пьёт пьёшь пьём пьёте льёт льёшь льём бьёт бьёшь бьём бьёте шьёт '
    'вьёт жуёт плюёт клюёт куёт берёт берёшь берём берёте зовёт зовёшь зовём зовёте ждёт ждёшь ждём ждёте '
    'врёт врёшь врём ревёт ревёшь плывёт плывёшь плывём несёт несёшь несём несёте везёт везёшь везём ведёт '
    'ведёшь ведём ведёте растёт растёшь растём трясёт метёт плетёт цветёт гребёт кладёт кладёшь кладём '
    'крадёт прядёт сечёт течёт печёт стережёт бережёт лжёт ржёт жжёт мнёт жмёт жнёт начнёт начнёшь начнём '
    'начнёте поймёт поймёшь поймём поймёте возьмёт возьмёшь возьмём возьмёте пошлёт пошлёшь пришлёт умрёт '
    'умрёшь умрём сотрёт оторвёт соврёт заснёт заснёшь уснёт уснёшь проснётся проснёшься вернётся вернёшься '
    'вернёмся вернётесь начнётся ведётся несётся зовётся берётся даётся смеётся смеёшься смеёмся смеётесь '
    'льётся бьётся рвётся остаётся остаёшься остаёмся остаётесь достаётся удаётся раздаётся подаётся '
    'продаётся передаётся сдаётся создаёт создаёшь создаём передаёт передаёшь передаём раздаёт подаёт '
    'продаёт издаёт отдаёт отдаёшь отдаём сдаёт задаёт задаёшь достаёт достаёшь '
    'собьёт собьёшь собьём убьёт убьёшь убьём разобьёт пробьёт добьёт прибьёт забьёт нальёт нальёшь прольёт '
    'польётся прольётся напьётся напьёшься споёт сошьёт займёт займёшь наймёт сожмёт пожмёт нажмёт прижмёт '
    'сожмётся отопрёт опрётся сгниёт кивнёт кивнёшь махнёт махнёшь шагнёт шагнёшь мелькнёт блеснёт рванёт '
    'толкнёт толкнёшь вздохнёт моргнёт свернёт развернёт завернёт шепнёт плеснёт глотнёт хлебнёт плюнёт '
    'улыбнётся улыбнёшься обернётся обернёшься повернётся повернёшься отвернётся отвернёшься сорвётся '
    'оборвётся порвётся ворвётся прорвётся соберётся соберёшься разберётся разберёшься наберётся заберётся '
    'доберётся доберёмся проберётся позовёт позовёшь назовёт назовёшь отзовётся подождёт подождём дождётся '
    'дождёмся принесёт принесёшь унесёт отнесёт занесёт спасёт спасёшь приведёт отведёт заведёт доведёт '
    'проведёт переведёт подведёт произведёт привезёт увезёт пойдёмте идёмте проживёт заживёт оживёт доживёт '
    'дёрнет дёрнешь дёрнул дёрнула дёрнули дёрнулся дёрнулась щёлкнет пришёлся сошёлся разошёлся обошёлся '
    'истёк протёк потёк утёк пёк испёк запёк чёлн счёты счётами ёрзал ёрзала ёрзали ёрзает ёкнуло ёкнуло '
    'ёжился ёжилась ёжась бельё жильё питьё житьё ружьё копьё сырьё старьё тряпьё вороньё зверьё враньё '
    'нытьё забытьё остриё'
).split()
_YO_PHRASES = [
    (r'\bвсе еще\b', 'всё ещё'), (r'\bвсе ещё\b', 'всё ещё'), (r'\bвсе равно\b', 'всё равно'),
    (r'\bвсе время\b', 'всё время'), (r'\bвот и все\b', 'вот и всё'),
    # after these prepositions "нем/чем/всем" can only be the prepositional "нём/чём/всём"
    (r'\b(о|об|обо|в|во|на|при)\s+нем\b', r'\1 нём'), (r'\b(о|об|обо|в|во|на|при)\s+чем\b', r'\1 чём'),
    (r'\b(обо|во|на|при)\s+всем\b', r'\1 всём'),
]


@lru_cache(maxsize=1)
def yo_dictionary():
    words = set(_YO_WORDS)
    paradigms = [
        (_YO_ADJ_HARD_STEMS, _HARD_ADJ), (_YO_ADJ_SOFT_STEMS, _SOFT_ADJ), (_YO_NOUN_M_STEMS, _NOUN_HARD_M),
        (_YO_NOUN_OK_STEMS, _NOUN_OK), (_YO_NOUN_SOFT_M_STEMS, _NOUN_SOFT_M), (_YO_NOUN_FEM_A_STEMS, _NOUN_FEM_A),
        (_YO_NOUN_FEM_SH_STEMS, _NOUN_FEM_SH), (_YO_NOUN_FEM_YA_STEMS, _NOUN_FEM_YA),
        (_YO_NOUN_FEM_KA_STEMS, _NOUN_FEM_KA), (_YO_ONOK_STEMS, _NOUN_ONOK),
    ]
    for stems, endings in paradigms:
        for stem in stems:
            words.update(stem + e for e in endings)
    mapping = {}
    for w in words:
        w = w.lower()
        if 'ё' in w:
            mapping[w.replace('ё', 'е')] = w
    return mapping


def _match_case(src, dst):
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


_CYR_WORD_RE = re.compile(r'[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*')


def restore_yo(text):
    mapping = yo_dictionary()

    def repl(m):
        w = m.group(0)
        if 'ё' in w or 'Ё' in w:
            return w
        target = mapping.get(w.lower())
        return _match_case(w, target) if target else w

    text = _CYR_WORD_RE.sub(repl, text)
    for pattern, replacement in _YO_PHRASES:
        if '\\1' in replacement:
            text = re.sub(pattern, lambda m, r=replacement: _match_case(m.group(0), m.expand(r)), text,
                          flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, lambda m, r=replacement: _match_case(m.group(0), r), text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------------------------------------------------
# Abbreviations, units, years
# ---------------------------------------------------------------------------------------------------------------------

_KEEP_DOT_RE = re.compile(r'\s*$|\s+[А-ЯЁA-Z«"„“—–]')


def _keep_sentence_dot(m, consumed_dot):
    """When an abbreviation's dot also ended the sentence ('... в 1834 г. Потом'), keep the dot.
    Not before a proper name ('в 1834 г. Пушкин написал'), where the sentence most likely continues."""
    if not consumed_dot or not _KEEP_DOT_RE.match(m.string, m.end()):
        return ''
    nxt = re.match(r'\s+([А-ЯЁ][а-яё]+)', m.string[m.end():])
    morph = _morph()
    if nxt and morph is not None:
        tag = morph.parse(nxt.group(1))[0].tag
        if any(t in tag for t in ('Name', 'Surn', 'Patr', 'Geox', 'Orgn')):
            return ''
    return '.'


_ABBREVIATIONS = [
    (r'\bт\.\s?е\.', 'то есть'), (r'\bт\.\s?к\.', 'так как'), (r'\bи\s+т\.\s?д\.', 'и так далее'),
    (r'\bи\s+т\.\s?п\.', 'и тому подобное'), (r'\bт\.\s?н\.', 'так называемый'), (r'\bи\s+др\.', 'и другие'),
    (r'\bи\s+пр\.', 'и прочее'), (r'\bдо\s+н\.\s?э\.', 'до нашей эры'), (r'\bн\.\s?э\.', 'нашей эры'),
    (r'\bнапр\.', 'например'), (r'\bсм\.', 'смотри'), (r'\bср\.(?=\s)', 'сравни'), (r'\b[гГ]-ну\b', 'господину'),
    (r'\b[гГ]-на\b', 'господина'), (r'\b[гГ]-ном\b', 'господином'), (r'\b[гГ]-не\b', 'господине'),
    (r'\b[гГ]-н\b', 'господин'), (r'\b[гГ]-жа\b', 'госпожа'), (r'\b[гГ]-жи\b', 'госпожи'), (r'\b[гГ]-же\b', 'госпоже'),
    (r'\b[гГ]-жу\b', 'госпожу'),
    (r'\bим\.(?=\s+[А-ЯЁ])', 'имени'), (r'\bпроф\.', 'профессор'), (r'\bакад\.', 'академик'),
    (r'\bтов\.(?=\s+[А-ЯЁ])', 'товарищ'), (r'\bP\.\s?S\.', 'постскриптум'), (r'\bок\.(?=\s+\d)', 'около'),
    (r'\bстр\.(?=\s*\d)', 'страница'), (r'\bгл\.(?=\s*[\dIVXLC])', 'глава'), (r'№\s*(?=\d)', 'номер '),
    (r'§\s*(?=\d)', 'параграф '), (r'(?<=\s)&(?=\s)', 'и'),
    # addresses ("г. Мурманск, ул. Книповича, д. 14, кв. 5"); years ("1834 г.") are handled before
    (r'(?<![\w.])г\.\s*(?=[А-ЯЁ][а-яё])', 'город '), (r'(?<![\w.])ул\.\s*(?=[А-ЯЁ])', 'улица '),
    (r'(?<![\w.])д\.\s*(?=\d)', 'дом '), (r'(?<![\w.])кв\.\s*(?=\d)', 'квартира '),
]


def expand_abbreviations(text):
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, lambda m, r=replacement: _match_case(m.group(0), r) +
                      _keep_sentence_dot(m, m.group(0).endswith('.')), text)
    return text


def _plural_form(n, forms):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


_UNITS = [  # (regex, forms for 1 / 2-4 / 5+, abbreviation may end with a dot)
    (r'%', ('процент', 'процента', 'процентов'), False),
    (r'руб\.?(?![а-яё])|р\.(?=\s|$)', ('рубль', 'рубля', 'рублей'), True),
    (r'коп\.?(?![а-яё])', ('копейка', 'копейки', 'копеек'), True),
    (r'км(?![а-яё])', ('километр', 'километра', 'километров'), False),
    (r'кг(?![а-яё])', ('килограмм', 'килограмма', 'килограммов'), False),
    (r'см(?![а-яё])', ('сантиметр', 'сантиметра', 'сантиметров'), False),
    (r'мм(?![а-яё])', ('миллиметр', 'миллиметра', 'миллиметров'), False),
    (r'тыс\.', ('тысяча', 'тысячи', 'тысяч'), True),
    (r'млн\.?(?![а-яё])', ('миллион', 'миллиона', 'миллионов'), True),
    (r'млрд\.?(?![а-яё])', ('миллиард', 'миллиарда', 'миллиардов'), True),
    (r'°C?(?![a-zA-Z])', ('градус', 'градуса', 'градусов'), False),
]
_CURRENCIES = {'$': ('доллар', 'доллара', 'долларов'), '€': ('евро', 'евро', 'евро'), '£': ('фунт', 'фунта', 'фунтов')}
_COMPOUND_SUFFIXES = (r'летн|лети|кратн|процентн|этажн|минутн|секундн|часов|дневн|недельн|месячн|градусн|тонн|'
                      r'метров|километров|рублёв|рублев|комнатн|томн|серийн|местн|ствольн|зарядн|дюймов|ярусн')


def postal_code_words(code):
    """'183038' -> 'сто восемьдесят три ноль тридцать восемь' (read in two groups of three digits)."""
    words = []
    for group in (code[:3], code[3:]):
        zeros = len(group) - len(group.lstrip('0'))
        rest = group.lstrip('0')
        words += ['ноль'] * zeros + ([cardinal(int(rest))] if rest else [])
    return ' '.join(words)


def _currency(sym, amount):
    forms = _CURRENCIES[sym]
    return forms[1] if re.search(r'[.,]', amount) else _plural_form(amount, forms)


def normalize_anonymized_years(text):
    """'18**', '18..' (anonymized years in classic literature) -> 'тысяча восемьсот…'"""
    return re.sub(r'\b(1[0-9])(?:\*{1,2}|\.\.(?!\.)|…)(?=[\s,.;:!?)»]|$)',
                  lambda m: _cap(m.string, m.start(), cardinal(int(m.group(1)) * 100)) + '…', text)


def normalize_units(text):
    """Spell out unit, currency, year and century abbreviations (the numbers themselves are converted later)."""
    # postal codes before a place: "183038, г. Мурманск" -> "сто восемьдесят три ноль тридцать восемь, г. Мурманск"
    text = re.sub(r'(?<![\d\w])(\d{6})(?=\s*,\s*(?:г\.|город|гор\.|пос\.|посёлок|с\.|д\.|ул\.|пгт|обл|край|респ))',
                  lambda m: _cap(m.string, m.start(), postal_code_words(m.group(1))), text)
    # arithmetic and phone numbers: "2+2", "+7 999" -> "2 плюс 2", "плюс 7 999"
    text = re.sub(r'(?<=\d)[ \t]*\+[ \t]*(?=\d)|(?<![\w+])\+(?=\d)', ' плюс ', text)
    # 1 000 000 -> 1000000
    text = re.sub(r'(?<![\d,.])(\d{1,3})((?:[   ]\d{3})+)(?!\d)',
                  lambda m: m.group(1) + re.sub(r'\s', '', m.group(2)), text)
    # dd.mm.yyyy -> "dd <month> yyyy года"
    text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',
                  lambda m: (f'{int(m.group(1))} {_MONTHS_GEN[int(m.group(2)) - 1]} {m.group(3)} года'
                             if 1 <= int(m.group(2)) <= 12 else m.group(0)), text)

    def year_abbr(m):  # "1834 г." -> "1834 года" / "в 1834 г." -> "в 1834 году"; "гг." -> годах/годов
        prep = _preceding_word(m.string, m.start())
        if m.group(2).startswith('гг'):
            word = 'годах' if prep in ('в', 'во', 'о', 'об', 'при') else 'годов'
        elif prep in ('в', 'во', 'о', 'об', 'при', 'к', 'по'):
            word = 'году'
        else:
            word = 'года'
        return m.group(1) + ' ' + word + _keep_sentence_dot(m, True)
    text = re.sub(r'(\d{3,4}(?:\s*[–—-]\s*\d{3,4})?)\s*(гг\.|г\.)(?![а-яё])', year_abbr, text)

    def century(m):  # "XIX в.", "XIX век", "в XIX в.", "XVIII—XIX вв."
        nums = [roman_to_int(p) for p in re.split(r'\s*[–—-]\s*', m.group(1))]
        if None in nums:
            return m.group(0)
        word, prep, dot = m.group(2), _preceding_word(m.string, m.start()), ''
        if word.startswith('вв'):
            word, dot = ('веках' if prep in ('в', 'во', 'о', 'при') else 'веков'), _keep_sentence_dot(m, True)
        elif word == 'в.':
            word = 'веке' if prep in ('в', 'во', 'о', 'при') else ('век' if prep in ('за', 'через') else 'века')
            dot = _keep_sentence_dot(m, True)
        return ' — '.join(str(n) for n in nums) + ' ' + word + dot
    text = re.sub(rf'(?<!\w)({_ROMAN_TOKEN}(?:\s*[–—-]\s*{_ROMAN_TOKEN})?)\s*(вв\.|в\.|век[а-я]*|столети[а-я]*)'
                  r'(?![а-яё])', century, text)

    for pattern, forms, has_dot in _UNITS:
        def unit(m, forms=forms, has_dot=has_dot):
            num = m.group(1)
            word = forms[1] if re.search(r'[.,]', num) else _plural_form(num, forms)
            return num + ' ' + word + _keep_sentence_dot(m, has_dot and m.group(0).endswith('.'))
        text = re.sub(rf'(\d+(?:[.,]\d+)?)\s*(?:{pattern})', unit, text)
    text = re.sub(r'([$€£])\s*(\d+(?:[.,]\d+)?)', lambda m: m.group(2) + ' ' + _currency(m.group(1), m.group(2)), text)
    text = re.sub(r'(\d+(?:[.,]\d+)?)\s*([$€£])', lambda m: m.group(1) + ' ' + _currency(m.group(2), m.group(1)), text)
    return text


def _time_words(h, mm):
    hours = cardinal(int(h))
    if mm == '00':
        return hours + ' ровно'
    if mm.startswith('0'):
        return hours + ' ноль ' + cardinal(int(mm))
    return hours + ' ' + cardinal(int(mm))


def _roman_after_word(m):
    """'Пётр I' -> 'Пётр Первый', 'Екатерины II' -> 'Екатерины Второй', 'Часть II' -> 'Часть вторая'."""
    word, space, roman = m.group(1), m.group(2), m.group(3)
    n = roman_to_int(roman)
    if n is None or (len(roman) == 1 and roman in 'CDML'):
        return m.group(0)
    nouns = _noun_parses(word.lower())
    if nouns:
        nn = nouns[0]
        res = ordinal(n, case=nn.case, gender=nn.gender, plural=nn.plural)
    elif word[:1].isupper():
        res = ordinal(n)
    else:
        return m.group(0)
    morph = _morph()
    if morph is not None and word[:1].isupper():
        tag = morph.parse(word)[0].tag
        if 'Name' in tag or 'Surn' in tag or 'Patr' in tag:
            res = res[:1].upper() + res[1:]
    return word + space + res


def _at_sentence_start(text, pos):
    before = text[:pos].rstrip(' \t«"„“(—–')
    return before == '' or before[-1] in '.!?…'


def _cap(text, pos, words):
    if words and _at_sentence_start(text, pos):
        return words[:1].upper() + words[1:]
    return words


def normalize_numbers(text):
    text = re.sub(rf'([А-ЯЁа-яё]+)([ \t]+)({_ROMAN_TOKEN})(?!\w)(?![.\-][А-ЯЁA-Z])', _roman_after_word, text)
    # Times: "10:30", "в 5.30"
    text = re.sub(r'(?<![\d.,:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])',
                  lambda m: _cap(m.string, m.start(), _time_words(m.group(1), m.group(2))), text)
    text = re.sub(r'(?<=\b[вк] )([01]?\d|2[0-3])\.([0-5]\d)(?![\d.,])', lambda m: _time_words(m.group(1), m.group(2)),
                  text)

    def compound(m):  # "20-летний" -> "двадцатилетний"
        n = int(m.group(1))
        prefix = {1: 'одно', 40: 'сорока', 90: 'девяносто', 100: 'сто', 1000: 'тысяче'}.get(n)
        return _cap(m.string, m.start(), (prefix or cardinal(n, case='g').replace(' ', '')) + m.group(2))
    text = re.sub(rf'(?<!\w)(\d+)-((?:{_COMPOUND_SUFFIXES})[а-яё]*)', compound, text)
    # Ranges "5-7" -> "5 — 7" (each number then agrees on its own)
    text = re.sub(r'(?<=\d)[ \t]*[–—][ \t]*(?=\d)|(?<=\d)-(?=\d)', ' — ', text)
    # Decimals
    text = re.sub(r'(?<![\d.,])(\d+,\d+)(?![.,]?\d)', lambda m: _cap(m.string, m.start(), _decimal_words(m.group(1))),
                  text)
    text = re.sub(r'(?<![\d.,])(\d+\.\d+)(?![.,]?\d)', lambda m: _cap(m.string, m.start(), _decimal_words(m.group(1))),
                  text)

    out, pos = [], 0
    for m in re.finditer(r'(?<!\w)(\d+)(?:-([а-яё]{1,3}))?(?!\w)', text):
        out.append(text[pos:m.start()])
        digits, suffix = m.group(1), m.group(2)
        if len(digits) > 15:  # phone numbers, codes...: digit by digit
            words = ' '.join(cardinal(int(d)) for d in digits)
        elif suffix and _is_colloquial_cardinal(int(digits), suffix):
            words = _colloquial_cardinal(int(digits), suffix, text, m.start(), m.end())
        else:
            words = number_words(int(digits), text, m.start(), m.end(), suffix=suffix)
        out.append(_cap(text, m.start(), words))
        pos = m.end()
    out.append(text[pos:])
    return ''.join(out)


# ---------------------------------------------------------------------------------------------------------------------
# Punctuation and headings
# ---------------------------------------------------------------------------------------------------------------------

_CAPS_RUN_RE = re.compile(r'(?<![\w-])[А-ЯЁ]+(?:-[А-ЯЁ]+)*(?:[ \t,;:—–«»"„“-]+[А-ЯЁ]+(?:-[А-ЯЁ]+)*)+(?![\w-])')
_CYR_VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')


def normalize_shouting(text, min_words=3):
    """
    'ОН НЕ ОЖИДАЛ, ЧТО ПОЛЮБИТ СВОЮ ЖЕНУ' -> 'Он не ожидал, что полюбит свою жену' (YouTube-style titles: TTS models
    may spell out capitalized words as abbreviations). Only runs of `min_words`+ capitalized words with real words
    in them; abbreviations (МВД, СССР, США) keep their capitals.
    """
    def word(w):
        w = w.group(0)
        if w in _KEEP_CAPS or not any(c in _CYR_VOWELS for c in w):
            return w
        return w.lower()

    def repl(m):
        run = m.group(0)
        words = re.findall(r'[А-ЯЁ]+(?:-[А-ЯЁ]+)*', run)
        if len(words) < min_words or sum(len(w) >= 4 and any(c in _CYR_VOWELS for c in w) for w in words) < 2:
            return run
        out = re.sub(r'[А-ЯЁ]+(?:-[А-ЯЁ]+)*', word, run)
        speech_start = re.search(r'[:—–«"„“]\s*$', m.string[:m.start()]) is not None
        if out[:1].islower() and (speech_start or _at_sentence_start(m.string, m.start())):
            out = out[:1].upper() + out[1:]
        return out
    return _CAPS_RUN_RE.sub(repl, text)


_KEEP_CAPS = {'США', 'ООН', 'ЕС', 'ОАЭ', 'ЮАР', 'КГБ', 'ЦРУ', 'ГУЛАГ', 'ЗАГС', 'ВУЗ', 'МГУ', 'АЭС', 'ГЭС', 'ОМОН',
              'ГИБДД', 'ГАИ', 'НКВД', 'ДНК', 'ИП', 'ООО', 'ОАО', 'ЗАО', 'ЖКХ', 'ЕГЭ', 'ОГЭ', 'НАТО', 'ЮНЕСКО'}


_LETTER_NAMES = {'Б': 'бэ', 'В': 'вэ', 'Г': 'гэ', 'Д': 'дэ', 'Ж': 'жэ', 'З': 'зэ', 'К': 'ка', 'Л': 'эль', 'М': 'эм',
                 'Н': 'эн', 'П': 'пэ', 'Р': 'эр', 'С': 'эс', 'Т': 'тэ', 'Ф': 'эф', 'Х': 'ха', 'Ц': 'цэ', 'Ч': 'че',
                 'Ш': 'ша', 'Щ': 'ща'}


def spell_consonant_abbreviations(text, stress=False):
    """
    'СРЗ' -> 'эс-эр-зэ', 'МВД' -> 'эм-вэ-дэ': an abbreviation without vowels can only be read by its letter names,
    which a TTS model may otherwise turn into an unpronounceable syllable. Abbreviations with vowels (США, БТИ, ВУЗ)
    are left as written (use --lexicon for the ones read by letters, e.g. "БТИ = бэ-тэ-и").
    """
    def repl(m):
        w = m.group(0)
        if re.fullmatch(_ROMAN_TOKEN, w):  # ХХ, ХIX (Roman numerals typed with Cyrillic letters)
            return w
        names = [_LETTER_NAMES[c] for c in w]
        if stress:  # letter abbreviations are stressed on their last letter: "эс-эр-зэ́"
            names[-1] = re.sub(r'([аэе])', '\\1\u0301', names[-1], count=1)
        return '-'.join(names)
    return re.sub(r'(?<![\w-])[БВГДЖЗКЛМНПРСТФХЦЧШЩ]{2,5}(?![\w-])', repl, text)


def strip_stress_marks(text):
    return text.replace('́', '').replace('̀', '')


def normalize_punctuation(text):
    text = re.sub(r'^\s*-{1,2}\s+', '— ', text)  # dialogue typed with a hyphen: "- Привет!"
    text = re.sub(r'(?<=\s)-{1,2}(?=\s)', '—', text)  # hyphen used as a dash
    text = re.sub(r'(?<=\S)[ \t]*([—–])[ \t]*(?=\S)', r' \1 ', text)  # "Германн,— сказал" -> "Германн, — сказал"
    text = re.sub(r'^\s*[—–]\s*', '— ', text)  # dialogue dash
    text = re.sub(r'([?!])\.\.(?!\.)', r'\1…', text)  # "?.." -> "?…"
    text = re.sub(r'\.{3,}', '…', text)
    text = re.sub(r'…\.+', '…', text)
    text = re.sub(r'([!?])\1+', r'\1', text)  # "!!!" -> "!"
    text = re.sub(r'[\[\]{}]', '', text)  # editorial brackets
    text = re.sub(r'[*_#|~^=\\<>]+', ' ', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'[ \t]+([,.;:!?…»)])', r'\1', text)
    text = re.sub(r'([«(])[ \t]+', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def ensure_terminal_punctuation(text):
    if text and re.search(r'[\w»"”’)]$', text) and not re.search(r'[.!?…:;][»"”’)]*$', text):
        return text + '.'
    return text


_HEADING_WORDS = {'глава': 'f', 'часть': 'f', 'книга': 'f', 'том': 'm', 'раздел': 'm'}


def normalize_heading(text):
    """'I' -> 'Глава первая', 'Глава 5. Возвращение' -> 'Глава пятая. Возвращение'."""
    t = text.strip()
    m = re.match(rf'^({_ROMAN_TOKEN}|\d{{1,3}})(?!\w)\s*\.?\s*(?:[.:—–-]\s*)?(.*)$', t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else roman_to_int(m.group(1))
        if n:
            rest = m.group(2).strip()
            return 'Глава ' + ordinal(n, gender='f') + ('. ' + rest if rest else '')
    m = re.match(rf'^(глава|часть|книга|том|раздел)\s+({_ROMAN_TOKEN}|\d{{1,3}})(?!\w)\s*\.?\s*(?:[.:—–-]\s*)?(.*)$',
                 t, re.IGNORECASE)
    if m:
        word, num, rest = m.group(1), m.group(2), m.group(3).strip()
        n = int(num) if num.isdigit() else roman_to_int(num)
        if n:
            head = word[:1].upper() + word[1:].lower() + ' ' + ordinal(n, gender=_HEADING_WORDS[word.lower()])
            return head + ('. ' + rest if rest else '')
    return t


# ---------------------------------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------------------------------

def normalize_ru(text, is_heading=False, keep_stress=False):
    if not keep_stress:
        text = strip_stress_marks(text)
    text = normalize_shouting(text, min_words=2 if is_heading else 3)
    if is_heading:
        text = normalize_heading(text)
    text = normalize_anonymized_years(text)
    text = normalize_punctuation(text)
    text = normalize_units(text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = spell_consonant_abbreviations(text, stress=keep_stress)
    text = restore_yo(text)
    text = normalize_punctuation(text)
    return ensure_terminal_punctuation(text)
