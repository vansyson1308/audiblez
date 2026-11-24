import re
import unittest

from audiblez.text import (BREAK, HEADING, PARA, Block, chunk_paragraph, html_to_blocks, is_chapter_heading_text,
                           split_long_text, split_sections, split_sentences, text_to_blocks)


def doc(body):
    return f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" ' \
           f'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>T</title><style>p{{}}</style></head>' \
           f'<body>{body}</body></html>'


class HtmlToBlocksTest(unittest.TestCase):
    def test_div_text_is_kept(self):
        blocks = html_to_blocks(doc('<div class="epigraph">Пиковая дама означает тайную недоброжелательность.</div>'
                                    '<p>Текст.</p>'))
        self.assertEqual([b.text for b in blocks], ['Пиковая дама означает тайную недоброжелательность.', 'Текст.'])

    def test_footnote_markers_and_bodies_removed(self):
        html = doc('<p>Увидеть la Vénus moscovite<sup class="reference"><a href="#n2" epub:type="noteref">[2]</a>'
                   '</sup>; Ришелье.</p><p>Он<a href="#n3">3</a> ушёл.</p>'
                   '<aside epub:type="footnote" id="n2"><p>московскую Венеру</p></aside>'
                   '<h2>Примечания</h2><ol><li>Впервые — в журнале.</li></ol>')
        texts = [b.text for b in html_to_blocks(html)]
        self.assertEqual(texts, ['Увидеть la Vénus moscovite; Ришелье.', 'Он ушёл.'])

    def test_notes_section_dropped_until_next_heading(self):
        html = doc('<h2>Глава 1</h2><p>Текст главы.</p><h2>Примечания</h2><p>Сноска 1.</p><p>Сноска 2.</p>'
                   '<h2>Глава 2</h2><p>Ещё текст.</p>')
        texts = [b.text for b in html_to_blocks(html)]
        self.assertEqual(texts, ['Глава 1', 'Текст главы.', 'Глава 2', 'Ещё текст.'])

    def test_br_lines_and_no_glued_words(self):
        blocks = html_to_blocks(doc('<p>А в ненастные дни<br/>Собирались они<br/>Часто;</p>'))
        self.assertEqual(blocks[0].text, 'А в ненастные дни, Собирались они, Часто;')

    def test_double_br_splits_paragraphs(self):
        blocks = html_to_blocks(doc('<p>Первый абзац.<br/><br/>Второй абзац.</p>'))
        self.assertEqual([b.text for b in blocks], ['Первый абзац.', 'Второй абзац.'])

    def test_nested_blocks_not_duplicated(self):
        blocks = html_to_blocks(doc('<ul><li><p>Пункт один.</p></li><li>Пункт два.</li></ul>'
                                    '<blockquote><p>Цитата.</p></blockquote>'))
        self.assertEqual([b.text for b in blocks], ['Пункт один.', 'Пункт два.', 'Цитата.'])

    def test_soft_hyphens_and_invisible_chars_removed(self):
        blocks = html_to_blocks(doc('<p>не­по­ко­ле­би­мым​ словом</p>'))
        self.assertEqual(blocks[0].text, 'непоколебимым словом')

    def test_scene_breaks_and_headings(self):
        blocks = html_to_blocks(doc('<h1>Книга</h1><p>Один.</p><p>* * *</p><p>Два.</p><hr/><p>Три.</p>'
                                    '<div style="text-align:center">IV</div><p>Четыре.</p>'))
        kinds = [(b.kind, b.text) for b in blocks]
        self.assertEqual(kinds, [(HEADING, 'Книга'), (PARA, 'Один.'), (BREAK, ''), (PARA, 'Два.'), (BREAK, ''),
                                 (PARA, 'Три.'), (HEADING, 'IV'), (PARA, 'Четыре.')])

    def test_scripts_ruby_and_nav_skipped(self):
        blocks = html_to_blocks(doc('<nav><ol><li>TOC</li></ol></nav><script>var x=1;</script>'
                                    '<p><ruby>漢<rt>kan</rt></ruby>字</p>'))
        self.assertEqual([b.text for b in blocks], ['漢字'])


class SectionsTest(unittest.TestCase):
    def test_split_document_with_several_chapters(self):
        long = 'Текст. ' * 100
        blocks = [Block(PARA, 'Эпиграф.'), Block(HEADING, 'I', 2), Block(PARA, long), Block(HEADING, 'II', 2),
                  Block(PARA, long), Block(HEADING, 'III', 2), Block(PARA, long)]
        sections = split_sections(blocks, default_title='Книга')
        self.assertEqual([s.title for s in sections], ['I', 'II', 'III'])
        self.assertEqual(sections[0].blocks[0].text, 'Эпиграф.')  # preamble goes with the first chapter

    def test_tiny_sections_are_merged(self):
        long = 'Текст. ' * 100
        blocks = [Block(HEADING, 'Часть первая', 1), Block(HEADING, 'Глава 1', 2), Block(PARA, long),
                  Block(HEADING, 'Глава 2', 2), Block(PARA, 'Коротко.'), Block(HEADING, 'Глава 3', 2), Block(PARA, long)]
        sections = split_sections(blocks)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, 'Часть первая. Глава 1')
        self.assertEqual(sections[1].title, 'Глава 2. Глава 3')

    def test_text_to_blocks_round_trip(self):
        blocks = text_to_blocks('Глава 1\nПервый абзац.\n\n* * *\nВторой абзац.\n')
        self.assertEqual([(b.kind, b.text) for b in blocks],
                         [(HEADING, 'Глава 1'), (PARA, 'Первый абзац.'), (BREAK, ''), (PARA, 'Второй абзац.')])


class HeadingDetectionTest(unittest.TestCase):
    def test_headings(self):
        for t in ['I', 'XII', 'ХII', 'Глава 5. Возвращение', 'Глава первая', 'ГЛАВА ПЯТАЯ', 'Пролог', 'Эпилог',
                  'Часть вторая', 'Chapter Twelve: The End', 'Chapter 3', '12']:
            self.assertTrue(is_chapter_heading_text(t), t)

    def test_not_headings(self):
        for t in ['Часть денег он отдал матери.', '— Как? что?', 'Глава семьи вошёл в комнату и сел.',
                  'Introduction to the problem of evil and other essays on many subjects', 'Вот и всё.']:
            self.assertFalse(is_chapter_heading_text(t), t)


class SentencesTest(unittest.TestCase):
    def test_russian_sentences(self):
        text = ('— Что ты сделал, Сурин? — спросил хозяин. Он ответил: «Проиграл!» А. С. Пушкин написал это. '
                'И ты ни разу не соблазнился? ни разу не поставил на руте?… Твёрдость твоя удивительна.')
        self.assertEqual(split_sentences(text, 'ru'), [
            '— Что ты сделал, Сурин? — спросил хозяин.',
            'Он ответил: «Проиграл!»',
            'А. С. Пушкин написал это.',
            'И ты ни разу не соблазнился? ни разу не поставил на руте?…',
            'Твёрдость твоя удивительна.'])

    def test_dialogue_continuation(self):
        text = 'Германн немец: он расчётлив, вот и всё! — заметил Томский. — А если кто для меня непонятен, так это.'
        self.assertEqual(split_sentences(text, 'ru'), [
            'Германн немец: он расчётлив, вот и всё! — заметил Томский.',
            '— А если кто для меня непонятен, так это.'])

    def test_english_abbreviations(self):
        self.assertEqual(split_sentences('Mr. Smith arrived. Dr. Who left.', 'en'),
                         ['Mr. Smith arrived.', 'Dr. Who left.'])

    def test_cjk(self):
        self.assertEqual(split_sentences('你好。我很好！你呢？', 'zh'), ['你好。', '我很好！', '你呢？'])


class ChunkingTest(unittest.TestCase):
    def test_chunks_respect_limit_and_keep_all_text(self):
        para = ('Он долго стоял у окна, глядя, как над рекой медленно гаснет закат; думал о том, что завтра всё '
                'изменится, что письмо придёт, что она вернётся — или не вернётся никогда. ') * 6
        for max_chars in (80, 150, 300):
            chunks = chunk_paragraph(para, 'ru', max_chars)
            self.assertTrue(all(len(c.text) <= max_chars for c in chunks), max_chars)
            joined = re.sub(r'\s+', ' ', ' '.join(c.text for c in chunks)).strip()
            self.assertEqual(joined, re.sub(r'\s+', ' ', para).strip())

    def test_sentences_are_packed_within_paragraph(self):
        chunks = chunk_paragraph('Да. Нет. Может быть. Конечно.', 'ru', 300)
        self.assertEqual([c.text for c in chunks], ['Да. Нет. Может быть. Конечно.'])
        self.assertTrue(chunks[0].sentence_end)

    def test_no_tiny_last_chunk(self):
        intro = ('Добро пожаловать, дорогие слушатели. Сегодня вас ждёт история, которая начинается в старом '
                 'деревянном доме на высоком волжском берегу, где осенью пахнет антоновкой и мокрыми досками. '
                 'Это рассказ о женщине, которая приехала продать материнский дом и случайно нашла за старым '
                 'буфетом двадцать семь неотправленных писем — и вместе с ними открыла человека, которого, как ей '
                 'казалось, знала всю жизнь. Устраивайтесь поудобнее, налейте себе чаю, если за окном тоже '
                 'непогода. Приятного прослушивания.')
        chunks = chunk_paragraph(intro, 'ru', 300)
        self.assertEqual(len(chunks), 3)  # greedy: 182 + 288 + 24 ("Приятного прослушивания." alone)
        self.assertTrue(all(len(c.text) <= 300 and c.sentence_end for c in chunks))
        self.assertTrue(all(len(c.text) >= 75 for c in chunks), [len(c.text) for c in chunks])
        self.assertEqual(' '.join(c.text for c in chunks), intro)
        self.assertTrue(chunks[-1].text.endswith('непогода. Приятного прослушивания.'))
        # paragraphs without a tiny chunk are packed as before (their cached audio stays valid)
        para = 'А' * 200 + '. ' + 'Б' * 150 + '.'
        self.assertEqual([len(c.text) for c in chunk_paragraph(para, 'ru', 300)], [201, 151])

    def test_long_sentence_split_at_clause(self):
        parts = split_long_text('первая часть фразы довольно длинная; вторая часть фразы тоже длинная', 50)
        self.assertEqual(parts, ['первая часть фразы довольно длинная;', 'вторая часть фразы тоже длинная'])


if __name__ == '__main__':
    unittest.main()
