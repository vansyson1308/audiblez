"""FB2 and DOCX inputs, and dialogue typed with hyphens in story scripts."""
import base64
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import audiblez.core as core
from audiblez.normalize_ru import normalize_ru
from audiblez.text import HEADING
from audiblez.textinput import markdown_to_blocks

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


class FormatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fb2(self):
        path = self.tmp / 'pushkin.fb2'
        path.write_text(FB2, encoding='utf-8')
        zpath = self.tmp / 'pushkin.fb2.zip'
        with zipfile.ZipFile(zpath, 'w') as z:
            z.write(path, 'pushkin.fb2')
        for p in (path, zpath):
            src = core.load_source(p)
            self.assertEqual((src.kind, src.title, src.creator, src.language),
                             ('fb2', 'Пиковая дама', 'Александр Сергеевич Пушкин', 'ru'))
            self.assertTrue(src.cover.startswith(b'\xff\xd8\xff'))
            self.assertEqual(len(src.documents), 1)  # the notes body is not read
            text = src.documents[0].extracted_text
            self.assertNotIn('Сноска', text)
            self.assertNotIn('[1]', text)
            self.assertNotIn('Александр Пушкин', text)  # title page of the body
            self.assertIn('Нарумова. Долгая', text)
            self.assertIn('А в ненастные дни, Собирались они', text)
            self.assertIn('* * *', text)  # scene break

    def test_fb2_pipeline(self):
        path = self.tmp / 'pushkin.fb2.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('pushkin.fb2', FB2)
        res = core.main(path, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True))
        self.assertEqual([c.title for c in res['chapters']], ['Глава I', 'Глава II'])
        first = [s.text for s in res['chapters'][0].segments]
        self.assertEqual(first[:2], ['Пиковая дама.', 'Александр Сергеевич Пушкин.'])  # announced from the metadata
        self.assertTrue(any(t.startswith('Пиковая дама означает') for t in first))  # epigraph kept in chapter 1
        self.assertTrue((self.tmp / 'out' / 'pushkin.audiblez' / 'script.txt').exists())

    def test_docx(self):
        path = self.tmp / 'story.docx'
        make_docx(path, [('a3', 'Свекровь'), ('Heading1', 'Глава 1'), (None, '- Ты кто такая? - спросила она.'),
                         (None, 'Строка стиха\nвторая строка'), ('Heading1', 'Глава 2'), (None, 'Конец.')])
        src = core.load_source(path)
        self.assertEqual((src.kind, src.title, src.language), ('text', 'Свекровь', 'ru'))
        blocks = src.documents[0].blocks
        self.assertEqual([b.text for b in blocks if b.kind == HEADING], ['Глава 1', 'Глава 2'])
        self.assertIn('Строка стиха, вторая строка', [b.text for b in blocks])
        res = core.main(path, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True))
        texts = [s.text for c in res['chapters'] for s in c.segments]
        self.assertIn('— Ты кто такая? — спросила она.', texts)

    def test_markdown_dialogue_is_not_a_list(self):
        blocks = markdown_to_blocks('- Ты кто такая? - спросила она.\n\n- пункт списка\n\n- Item\n')
        self.assertEqual([b.text for b in blocks], ['— Ты кто такая? - спросила она.', 'пункт списка', 'Item'])

    def test_hyphen_dialogue_normalization(self):
        self.assertEqual(normalize_ru('- Привет, - сказал он. -- Ну и ну.'), '— Привет, — сказал он. — Ну и ну.')
        self.assertEqual(normalize_ru('Северо-западный ветер.'), 'Северо-западный ветер.')


if __name__ == '__main__':
    unittest.main()
