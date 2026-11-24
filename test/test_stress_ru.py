"""Russian stress marks (--stress ruaccent) and all-caps normalization. RUAccent itself is replaced by a fake."""
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import audiblez.core as core
from audiblez.audio import char_error_rate
from audiblez.normalize_ru import normalize_ru
from audiblez.stress_ru import (ACUTE, STRESS_MODEL, Stressor, lower_title_case, marks_to_plus, plus_to_acute,
                                protect_plus_signs, restore_case)

DICT = {'замок': 'зам+ок', 'стоит': 'сто+ит', 'какая': 'как+ая', 'сегодня': 'сег+одня', 'погода': 'пог+ода',
        'все': 'вс+е', 'ее': 'её', 'мысли': 'м+ысли', 'были': 'б+ыли', 'о': 'о', 'детях': 'д+етях',
        'германн': 'г+ерманн', 'сидел': 'сид+ел', 'молча': 'м+олча'}


class FakeAccentizer:
    """Lowercase-only dictionary lookup, like RUAccent: '+' before the stressed vowel, е -> ё."""

    def __init__(self):
        self.calls = []

    def process_all(self, text):
        self.calls.append(text)

        def repl(m):
            w = m.group(0)
            if '+' in w:
                return w
            return DICT.get(w, w) if w.islower() else w
        return re.sub(r'[а-яёА-ЯЁ+]+', repl, text)


class StressHelpersTest(unittest.TestCase):
    def test_mark_conversions(self):
        self.assertEqual(marks_to_plus('за́мок и до̀м'), 'з+амок и дом')
        self.assertEqual(plus_to_acute('з+амок сто+ит'), 'за́мок стои́т')
        self.assertEqual(plus_to_acute('з+амок'), 'за' + ACUTE + 'мок')
        self.assertEqual(protect_plus_signs('два+два и з+амок, C++'), 'два плюс два и з+амок, C плюс  плюс ')

    def test_lower_title_case(self):
        self.assertEqual(lower_title_case('Какая погода? Потом СССР и Я.'), 'какая погода? потом СССР и Я.')
        self.assertEqual(lower_title_case('Г+ерманн'), 'г+ерманн')

    def test_restore_case(self):
        self.assertEqual(restore_case('Все ее мысли', 'вс+е её м+ысли'), 'Вс+е её м+ысли')
        self.assertEqual(restore_case('ВСЕ', 'вс+ё'), 'ВС+Ё')
        self.assertIsNone(restore_case('Все', 'кот'))
        self.assertIsNone(restore_case('Все её', 'вс+е'))

    def test_stressor_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakeAccentizer()
            st = Stressor(Path(d) / 'stress.json', accentizer=fake)
            out = st('Какая сегодня погода? Все ее мысли были о детях.')
            self.assertEqual(out, 'Кака́я сего́дня пого́да? Все́ её мы́сли бы́ли о де́тях.')
            self.assertEqual(fake.calls, ['какая сегодня погода? все ее мысли были о детях.'])  # lowercased input
            self.assertEqual(st('Ге́рманн сидел молча.'), 'Ге́рманн сиде́л мо́лча.')  # existing mark kept
            st.save()
            st2 = Stressor(Path(d) / 'stress.json', accentizer=FakeAccentizer())
            st2('Какая сегодня погода? Все ее мысли были о детях.')
            self.assertEqual(st2.accentizer.calls, [])  # from the cache

    def test_corrections_after_ruaccent(self):
        from audiblez.stress_ru import BY_PREPOSITION, NORMATIVE, apply_corrections
        for word, value in list(NORMATIVE.items()) + [(w, v) for w, (_, a, b) in BY_PREPOSITION.items() for v in (a, b)]:
            self.assertEqual(value.replace('+', ''), word)
            self.assertEqual(value.count('+'), 1)
            self.assertIn(value[value.index('+') + 1], 'аеёиоуыэюя')
        src = 'вдруг поняла, у самой двери, вытирая руки, ледоход начался, из руки, стоял в двери, ПОНЯЛА, Г+убы'
        out = 'вдруг п+оняла, у с+амой двер+и, вытир+ая рук+и, ледох+од н+ачался, из р+уки, сто+ял в дв+ери, П+ОНЯЛА, Г+убы'
        self.assertEqual(plus_to_acute(apply_corrections(src, out)),
                         'вдруг поняла́, у са́мой две́ри, вытира́я ру́ки, ледохо́д начался́, из руки́, стоя́л в двери́, '
                         'ПОНЯЛА́, Гу́бы')  # a mark that was already in the text (lexicon) is kept
        self.assertEqual(apply_corrections('один два', 'од+ин'), 'од+ин')  # words do not line up: unchanged

    def test_cer_ignores_stress_marks(self):
        self.assertEqual(char_error_rate('Ве́чером за́мок опусте́л.', 'Вечером замок опустел'), 0.0)


class ShoutingTest(unittest.TestCase):
    def test_caps_titles(self):
        self.assertEqual(normalize_ru('ОН НЕ ОЖИДАЛ, ЧТО ПОЛЮБИТ СВОЮ ЖЕНУ'), 'Он не ожидал, что полюбит свою жену.')
        self.assertEqual(normalize_ru('РАСПЛАТА ЗА ИЗМЕНУ. Очень трогательная история.'),
                         'Расплата за измену. Очень трогательная история.')
        self.assertEqual(normalize_ru('Она крикнула: — Я НЕ ВЕРНУСЬ НИКОГДА!'), 'Она крикнула: — Я не вернусь никогда!')
        self.assertEqual(normalize_ru('и тогда Я ПОНЯЛА ВСЁ СРАЗУ.'), 'и тогда я поняла всё сразу.')
        self.assertEqual(normalize_ru('ГЛАВА ПЕРВАЯ', is_heading=True), 'Глава первая.')

    def test_abbreviations_untouched(self):
        # kept as abbreviations (not lowercased), then read by their letter names
        self.assertEqual(normalize_ru('Сотрудники МВД и ФСБ приехали.'), 'Сотрудники эм-вэ-дэ и эф-эс-бэ приехали.')
        self.assertEqual(normalize_ru('После ЗАГС ОНИ ПОШЛИ ДОМОЙ.'), 'После ЗАГС они пошли домой.')
        self.assertEqual(normalize_ru('В СССР и США.'), 'В эс-эс-эс-эр и США.')

    def test_letter_abbreviations_and_lexicon_case(self):
        self.assertEqual(normalize_ru('Тридцать пятый СРЗ.', keep_stress=True), 'Тридцать пятый эс-эр-зэ́.')
        from audiblez.normalize import Lexicon
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'lex.txt'
            path.write_text('БТИ = бэ-тэ-и\nГерманн = Герман\n', encoding='utf-8')
            lex = Lexicon.load(path)
            self.assertEqual(lex.apply('Старый план БТИ. БТИ закрыли. Германн пришёл.'),
                             'Старый план бэ-тэ-и. Бэ-тэ-и закрыли. Герман пришёл.')

    def test_plus_sign(self):
        self.assertEqual(normalize_ru('Было 2+2.'), 'Было два плюс два.')


class StressPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = Stressor.accentizer
        Stressor.accentizer = property(lambda self: self.__dict__.setdefault('_fake', FakeAccentizer()))

    def tearDown(self):
        Stressor.accentizer = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_with_stress(self):
        story = self.tmp / 'story.txt'
        story.write_text('Германн сидел молча. Какая сегодня погода?\n', encoding='utf-8')
        settings = core.Settings(dry_run=True, stress='ruaccent', announce_title=False)
        res = core.main(story, output_folder=self.tmp, settings=settings)
        self.assertEqual(res['chapters'][0].segments[0].text, 'Ге́рманн сиде́л мо́лча. Кака́я сего́дня пого́да?')
        self.assertTrue((self.tmp / 'story.audiblez' / 'stress_cache.json').exists())
        choice = core.resolve_engine_choice(settings, 'ru')
        self.assertEqual(core.with_stress_model(settings, choice).qwen_model, STRESS_MODEL)
        self.assertIsNone(core.with_stress_model(core.Settings(), choice).qwen_model)
        explicit = core.Settings(stress='ruaccent', qwen_model='my/model')
        self.assertEqual(core.with_stress_model(explicit, choice).qwen_model, 'my/model')
        ja = core.resolve_engine_choice(core.Settings(stress='ruaccent'), 'ja')
        self.assertIsNone(core.with_stress_model(settings, ja).qwen_model)  # Russian only
        _, fingerprint = core.engine_fingerprint(choice, core.with_stress_model(settings, choice))
        self.assertIn(STRESS_MODEL, fingerprint)  # a separate cache from the base model


if __name__ == '__main__':
    unittest.main()
