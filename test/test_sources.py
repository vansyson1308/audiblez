"""Story scripts (.txt/.md), FB2 and DOCX inputs."""
import base64
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import audiblez.core as core
from audiblez.normalize_ru import normalize_ru
from audiblez.sources import decode_text, detect_language, read_source, text_to_source_blocks
from audiblez.text import BREAK, HEADING, PARA

STORY = '''Брак на спор

Глава 1

Татьяна стояла у окна и смотрела на дождь. Ей было 35 лет.
- Ты опять задержался, - тихо сказала она.
- Работа, - буркнул муж и прошёл мимо.

* * *

Через неделю всё изменилось. Татьяна собрала вещи, сняла маленькую квартиру на окраине города и впервые за много лет почувствовала, что может дышать свободно. Она устроилась на работу в библиотеку, по вечерам гуляла в парке и училась жить заново, без оглядки на чужое мнение.

Глава 2

Утром позвонила свекровь. Голос у неё был холодный и требовательный, как всегда: она хотела знать, когда невестка вернётся домой и перестанет позорить семью. Татьяна молча выслушала и положила трубку. Потом долго сидела на кухне с остывшим чаем и думала о том, сколько лет потратила на людей, которые никогда её не ценили.
'''

WRAPPED = '''Однажды играли в карты у конногвардейца Нарумова. Долгая зимняя ночь
прошла незаметно; сели ужинать в пятом часу утра. Те, которые остались в
выигрыше, ели с большим аппетитом; прочие, в рассеянности, сидели перед
пустыми своими приборами.
— Что ты сделал, Сурин? — спросил хозяин.
— Проиграл, по обыкновению.

Но шампанское явилось, разговор оживился, и все приняли в нём участие.
'''

FB2 = '''<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
 <description><title-info>
   <author><first-name>Александр</first-name><middle-name>Сергеевич</middle-name><last-name>Пушкин</last-name></author>
   <book-title>Пиковая дама</book-title><lang>ru</lang>
   <coverpage><image l:href="#cover.jpg"/></coverpage>
 </title-info></description>
 <body>
  <title><p>Александр Пушкин</p><p>Пиковая дама</p></title>
  <epigraph><p>Пиковая дама означает тайную недоброжелательность.</p></epigraph>
  <section>
   <title><p>Глава I</p></title>
   <p>Однажды играли в карты у конногвардейца Нарумова<a l:href="#n1" type="note">[1]</a>. Долгая зимняя ночь прошла
   незаметно; сели ужинать в пятом часу утра. Те, которые остались в выигрыше, ели с большим аппетитом.</p>
   <empty-line/>
   <poem><stanza><v>А в ненастные дни</v><v>Собирались они</v></stanza></poem>
   <subtitle>* * *</subtitle>
   <p>Но шампанское явилось, разговор оживился, и все приняли в нём участие. Германн сидел молча и смотрел.</p>
  </section>
  <section>
   <title><p>Глава II</p></title>
   <p>Старая графиня *** сидела в своей уборной перед зеркалом. Три девушки окружали её, одна держала банку румян.
   Графиня не имела никакой претензии на красоту, давно увядшую, но сохраняла все привычки своей молодости, строго следовала модам семидесятых годов и одевалась так же долго, так же старательно, как и шестьдесят лет тому назад.</p>
  </section>
 </body>
 <body name="notes"><section id="n1"><title><p>1</p></title><p>Сноска о Нарумове.</p></section></body>
 <binary id="cover.jpg" content-type="image/jpeg">''' + base64.b64encode(b'\xff\xd8\xff\xe0fakejpeg').decode() + '''</binary>
</FictionBook>
'''


def make_docx(path, paragraphs):
    """Minimal .docx: paragraphs = [(style_id or None, text)]; Heading1 / Title styles declared in styles.xml."""
    w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    body = ''
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        runs = ''.join(f'<w:r><w:t xml:space="preserve">{t}</w:t></w:r>' if i == 0 else
                       f'<w:r><w:br/><w:t xml:space="preserve">{t}</w:t></w:r>' for i, t in enumerate(text.split('\n')))
        body += f'<w:p>{ppr}{runs}</w:p>'
    styles = (f'<w:styles xmlns:w="{w}"><w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
              f'<w:style w:styleId="a3"><w:name w:val="Title"/></w:style></w:styles>')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', '<Types/>')
        z.writestr('word/document.xml', f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>')
        z.writestr('word/styles.xml', styles)


class SourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, content, encoding='utf-8'):
        path = self.tmp / name
        path.write_bytes(content.encode(encoding))
        return path

    def test_story_txt(self):
        src = read_source(self.write('story.txt', STORY))
        self.assertEqual(src.title, 'Брак на спор')
        self.assertEqual(src.language, 'ru')
        blocks = src.documents[0].blocks
        self.assertEqual([(b.kind, b.text) for b in blocks if b.kind == HEADING], [(HEADING, 'Глава 1'),
                                                                                  (HEADING, 'Глава 2')])
        paras = [b.text for b in blocks if b.kind == PARA]
        self.assertIn('— Ты опять задержался, — тихо сказала она.', paras)  # hyphens typed as dialogue dashes
        self.assertIn(BREAK, [b.kind for b in blocks])
        self.assertEqual(len(paras), 5)  # one line = one paragraph

    def test_hard_wrapped_txt(self):
        blocks = text_to_source_blocks(WRAPPED)
        paras = [b.text for b in blocks if b.kind == PARA]
        self.assertEqual(len(paras), 4, paras)
        self.assertTrue(paras[0].startswith('Однажды играли') and paras[0].endswith('пустыми своими приборами.'))
        self.assertIn('ночь прошла незаметно', paras[0])
        self.assertEqual(paras[1], '— Что ты сделал, Сурин? — спросил хозяин.')

    def test_markdown(self):
        md = ('# Тайна старого дома\n\n## Глава 1. Письмо\n\n**Марина** открыла [письмо](http://x.ru) и *замерла*.\n\n'
              '---\n\n- Кто это написал? - спросила она.\n')
        src = read_source(self.write('story.md', md))
        self.assertEqual(src.title, 'Тайна старого дома')
        blocks = src.documents[0].blocks
        self.assertEqual((blocks[0].kind, blocks[0].text), (HEADING, 'Глава 1. Письмо'))
        self.assertEqual(blocks[1].text, 'Марина открыла письмо и замерла.')
        self.assertEqual(blocks[2].kind, BREAK)
        self.assertEqual(blocks[3].text, '— Кто это написал? — спросила она.')

    def test_cp1251_and_broken_utf8(self):
        self.assertEqual(decode_text('Привет, мир'.encode('cp1251')), 'Привет, мир')
        self.assertEqual(decode_text('﻿Привет'.encode('utf-8')), 'Привет')
        text = ('Это длинный текст. ' * 300).encode('utf-8') + b'\xff' + 'конец'.encode('utf-8')
        self.assertTrue(decode_text(text).endswith('конец'))
        src = read_source(self.write('legacy.txt', STORY, encoding='cp1251'))
        self.assertEqual(src.title, 'Брак на спор')

    def test_detect_language(self):
        self.assertEqual(detect_language('Однажды играли в карты.'), 'ru')
        self.assertEqual(detect_language('Це була довга і тиха ніч, і всі спали.'), 'uk')
        self.assertEqual(detect_language('It was a bright cold day.'), 'en')
        self.assertEqual(detect_language('吾輩は猫である。名前はまだ無い。'), 'ja')
        self.assertEqual(detect_language('옛날 옛적에 호랑이가 살았습니다.'), 'ko')
        self.assertIsNone(detect_language('123 456'))

    def test_fb2(self):
        path = self.write('pushkin.fb2', FB2)
        for p in (path, self._zip(path)):
            src = read_source(p)
            self.assertEqual(src.title, 'Пиковая дама')
            self.assertEqual(src.creator, 'Александр Сергеевич Пушкин')
            self.assertEqual(src.language, 'ru')
            self.assertTrue(src.cover.startswith(b'\xff\xd8\xff'))
            self.assertEqual(len(src.documents), 1)  # the notes body is not read
            text = src.documents[0].extracted_text
            self.assertNotIn('Сноска', text)
            self.assertNotIn('[1]', text)
            self.assertNotIn('Александр Пушкин', text)  # body title page
            self.assertIn('Нарумова. Долгая', text)
            self.assertIn('А в ненастные дни, Собирались они', text)
            self.assertIn('* * *', text)  # scene break
            sections = core.document_sections(src.documents[0])
            self.assertEqual([s.title for s in sections], ['Глава I', 'Глава II'])
            self.assertTrue(sections[0].blocks[0].text.startswith('Пиковая дама означает'))  # epigraph kept

    def _zip(self, path):
        zpath = self.tmp / 'pushkin.fb2.zip'
        with zipfile.ZipFile(zpath, 'w') as z:
            z.write(path, 'pushkin.fb2')
        return zpath

    def test_docx(self):
        path = self.tmp / 'story.docx'
        make_docx(path, [('a3', 'Свекровь'), ('Heading1', 'Глава 1'), (None, '- Ты кто такая? - спросила она.'),
                         (None, 'Строка стиха\nвторая строка'), ('Heading1', 'Глава 2'), (None, 'Конец.')])
        src = read_source(path)
        self.assertEqual(src.title, 'Свекровь')
        blocks = src.documents[0].blocks
        self.assertEqual([b.text for b in blocks if b.kind == HEADING], ['Глава 1', 'Глава 2'])
        self.assertIn('— Ты кто такая? — спросила она.', [b.text for b in blocks])
        self.assertIn('Строка стиха, вторая строка', [b.text for b in blocks])

    def test_pipeline_dry_run_on_txt(self):
        path = self.write('Свекровь.txt', STORY)
        res = core.main(path, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True))
        chapters = res['chapters']
        self.assertEqual([c.title for c in chapters], ['Глава 1', 'Глава 2'])
        texts = [s.text for s in chapters[0].segments]
        self.assertEqual(texts[0], 'Брак на спор.')  # the title is announced once
        self.assertEqual(texts[1], 'Глава первая.')
        self.assertIn('Ей было тридцать пять лет.', ' '.join(texts))
        self.assertTrue((self.tmp / 'out' / 'Свекровь.audiblez' / 'script.txt').exists())

    def test_hyphen_dialogue_normalization(self):
        self.assertEqual(normalize_ru('- Привет, - сказал он. -- Ну и ну.'), '— Привет, — сказал он. — Ну и ну.')
        self.assertEqual(normalize_ru('Северо-западный ветер.'), 'Северо-западный ветер.')


if __name__ == '__main__':
    unittest.main()
