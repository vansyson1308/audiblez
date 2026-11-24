"""Japanese / Korean text input and normalization, and original-content (.txt/.md) sources."""
import shutil
import tempfile
import unittest
from pathlib import Path

import audiblez.core as core
from audiblez.normalize import Lexicon, normalize_text
from audiblez.text import (HEADING, chunk_paragraph, detect_language, html_to_blocks, is_chapter_heading_text,
                           split_sentences)
from audiblez.textinput import convert_ruby, read_text_book, read_text_file


class JapaneseTextTest(unittest.TestCase):
    def test_sentences_keep_quotation_with_to(self):
        text = '静かな夜だった。「本当に、彼は帰ってくると思う？」と彼女は尋ねた。僕は何も答えられなかった……。窓の外は白み始めていた。'
        self.assertEqual(split_sentences(text, 'ja'), [
            '静かな夜だった。', '「本当に、彼は帰ってくると思う？」と彼女は尋ねた。', '僕は何も答えられなかった……。',
            '窓の外は白み始めていた。'])

    def test_chunks_are_short_enough(self):
        para = '静かな夜だった。' * 40
        chunks = chunk_paragraph(normalize_text(para, 'ja'), 'ja', 110)
        self.assertTrue(all(len(c.text) <= 110 for c in chunks))
        self.assertEqual(''.join(c.text for c in chunks), para)

    def test_ruby_notation(self):
        self.assertEqual(convert_ruby('｜颯《はやて》は東京（とうきょう）へ。《《本当に》》行くのか［＃「行く」に傍点］'),
                         'はやてはとうきょうへ。本当に行くのか')
        self.assertEqual(convert_ruby('魔王城《まおうじょう》に着いた'), 'まおうじょうに着いた')
        self.assertEqual(convert_ruby('彼（かれ）と猫(ねこ)'), 'かれとねこ')
        self.assertEqual(convert_ruby('普通の括弧（注釈です）は残る'), '普通の括弧（注釈です）は残る')
        self.assertEqual(convert_ruby('｜颯《はやて》', mode='base'), '颯')

    def test_epub_ruby_reading(self):
        html = '<p><ruby>颯<rt>はやて</rt></ruby>は<ruby>東京<rp>(</rp><rt>とうきょう</rt><rp>)</rp></ruby>へ。</p>'
        self.assertEqual(html_to_blocks(html, ruby='reading')[0].text, 'はやてはとうきょうへ。')
        self.assertEqual(html_to_blocks(html)[0].text, '颯は東京へ。')

    def test_normalization(self):
        cases = [('「待って！！」', '「待って！」'), ('あ〜、そうか……', 'あー、そうか……'),
                 ('【鑑定】スキルを使った', '鑑定スキルを使った。'), ('２０２６年、ＡＩが　世界を変えた', '2026年、AIが世界を変えた。'),
                 ('それは――', 'それは……'), ('ｶﾞｯｺｳに行く', 'ガッコウに行く。'), ('星が綺麗だ♪', '星が綺麗だ。')]
        for src, expected in cases:
            self.assertEqual(normalize_text(src, 'ja'), expected, src)
        self.assertEqual(normalize_text('第一章　旅立ち', 'ja', is_heading=True), '第一章、旅立ち。')

    def test_lexicon_matches_inside_japanese_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'lex.txt'
            p.write_text('颯 = はやて\n魔王城 = まおうじょう\n', encoding='utf-8')
            self.assertEqual(normalize_text('颯は魔王城へ向かった', 'ja', lexicon=Lexicon.load(p)),
                             'はやてはまおうじょうへ向かった。')


class KoreanTextTest(unittest.TestCase):
    def test_sentences_split_on_periods(self):
        text = '방 안은 너무 조용했다. "정말 그가 돌아올까?" 그녀가 물었다. 나는 대답하지 못했다.'
        self.assertEqual(split_sentences(text, 'ko'),
                         ['방 안은 너무 조용했다.', '"정말 그가 돌아올까?"', '그녀가 물었다.', '나는 대답하지 못했다.'])

    def test_normalization(self):
        self.assertEqual(normalize_text('비가 내리고 있었다', 'ko'), '비가 내리고 있었다.')
        self.assertEqual(normalize_text('그래... 알았어!!', 'ko'), '그래… 알았어!')
        self.assertEqual(normalize_text('３명이 왔다', 'ko'), '세 명이 왔다.')
        self.assertEqual(normalize_text('５０％ 할인', 'ko'), '오십 퍼센트 할인.')
        self.assertEqual(normalize_text('3장 그날', 'ko', is_heading=True), '삼 장 그날.')

    def test_numbers_native_and_sino_korean(self):
        from audiblez.normalize_ko import normalize_numbers_ko as n
        cases = [('3명이 왔다', '세 명이 왔다'), ('2026년 6월 10일', '이천이십육 년 유월 십 일'), ('20살', '스무 살'),
                 ('21살', '스물한 살'), ('오후 3시에', '오후 세 시에'), ('10:30에 출발', '열 시 삼십 분에 출발'),
                 ('1시간 30분', '한 시간 삼십 분'), ('5,000원', '오천 원'), ('50%', '오십 퍼센트'),
                 ('3.5킬로미터', '삼 점 오 킬로미터'), ('3번째 방', '세 번째 방'), ('10월 9일', '시월 구 일'),
                 ('12층', '십이 층'), ('15000원', '만 오천 원'), ('123456789', '일억 이천삼백사십오만 육천칠백팔십구'),
                 ('2개월', '두 개월'), ('99마리', '아흔아홉 마리'), ('제3장', '제삼 장'), ('1990년대', '천구백구십 년대')]
        for src, expected in cases:
            self.assertEqual(n(src), expected, src)
        self.assertEqual(normalize_text('"정말?"', 'ko'), '"정말?"')


class HeadingsAndLanguageTest(unittest.TestCase):
    def test_cjk_headings(self):
        for t in ['第一章　旅立ち', '第1話「出会い」', 'プロローグ', '第十二話', '제1장 시작', '12화', '에필로그', '番外編']:
            self.assertTrue(is_chapter_heading_text(t), t)
        for t in ['第三章はここで終わる、と彼は言ったが、それは本当なのだろうか。誰も知らない長い話だ。', '3장에서 말했듯이 우리는 떠났다']:
            self.assertFalse(is_chapter_heading_text(t), t)

    def test_detect_language(self):
        self.assertEqual(detect_language('静かな夜だった。'), 'ja')
        self.assertEqual(detect_language('방 안은 조용했다'), 'ko')
        self.assertEqual(detect_language('他说我们明天一起去看海'), 'zh')
        self.assertEqual(detect_language('Тишина'), 'ru')
        self.assertEqual(detect_language('Hello world'), 'en')


class TextInputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_legacy_encodings(self):
        samples = {'ja': ('第一章　旅立ち\n　「待って！」と彼女は叫んだ。\n', ['cp932', 'euc-jp']),
                   'ko': ('제1장 시작\n방 안은 조용했다. "정말 그가 돌아올까?"\n', ['cp949']),
                   'zh': ('第一章\n他说我们明天一起去看海，你也来吧。\n', ['gb18030']),
                   'ru': ('Глава 1\nТишина стояла такая, что было слышно часы.\n', ['cp1251', 'koi8-r'])}
        for lang, (text, encodings) in samples.items():
            for enc in encodings:
                f = self.tmp / f'{lang}_{enc}.txt'
                f.write_bytes(text.encode(enc))
                self.assertEqual(read_text_file(f), text, f'{lang} {enc}')

    def test_japanese_web_novel_txt(self):
        f = self.tmp / '星の旅人.txt'
        f.write_bytes(('第一章　旅立ち\n\n　｜颯《はやて》は東京（とうきょう）へ向かった。\n◇◇◇\n「待って！」と彼女は叫んだ。\n'
                       '第二章　再会\n　雨が降っていた。\n').encode('cp932'))
        meta, docs = read_text_book(f)
        self.assertEqual(meta['language'], 'ja')
        self.assertEqual(meta['title'], '星の旅人')
        kinds = [(b.kind, b.text) for b in docs[0].blocks]
        self.assertEqual(kinds, [('heading', '第一章 旅立ち'), ('para', 'はやてはとうきょうへ向かった。'), ('break', ''),
                                 ('para', '「待って！」と彼女は叫んだ。'), ('heading', '第二章 再会'), ('para', '雨が降っていた。')])

    def test_markdown_with_front_matter(self):
        f = self.tmp / 'book.md'
        f.write_text('---\ntitle: 星の旅人\nauthor: 山田\nlanguage: ja\n---\n# 第1話 出会い\n\n雨が降っていた。\n**彼**は傘を\n'
                     '持っていなかった。\n\n---\n\n## 第2話\n\n翌朝。\n', encoding='utf-8')
        meta, docs = read_text_book(f)
        self.assertEqual((meta['title'], meta['author'], meta['language']), ('星の旅人', '山田', 'ja'))
        self.assertEqual([(b.kind, b.text) for b in docs[0].blocks],
                         [('heading', '第1話 出会い'), ('para', '雨が降っていた。彼は傘を持っていなかった。'), ('break', ''),
                          ('heading', '第2話'), ('para', '翌朝。')])

    def test_korean_lines_keep_spaces(self):
        f = self.tmp / 'k.md'
        f.write_text('# 제1장\n\n비가 내리고\n있었다. 그는 우산이\n없었다.\n', encoding='utf-8')
        self.assertEqual(read_text_book(f)[1][0].blocks[-1].text, '비가 내리고 있었다. 그는 우산이 없었다.')

    def test_hard_wrapped_english(self):
        f = self.tmp / 'en.txt'
        f.write_text('Chapter 1\n\nIt was a dark and stormy night; the rain fell in\ntorrents, except at occasional '
                     'intervals, when it\nwas checked by a violent gust of wind.\n\nThe second paragraph starts here and it\n'
                     'continues on this line.\n', encoding='utf-8')
        blocks = read_text_book(f)[1][0].blocks
        self.assertEqual([b.kind for b in blocks], [HEADING, 'para', 'para'])
        self.assertTrue(blocks[1].text.endswith('violent gust of wind.'))

    def test_folder_of_chapters_in_natural_order(self):
        d = self.tmp / 'novel'
        d.mkdir()
        for i in (1, 2, 10):
            (d / f'第{i}話.txt').write_text(f'第{i}話\n本文{i}です。\n', encoding='utf-8')
        (d / 'cover.png').write_bytes(b'\x89PNG\r\n\x1a\n')
        meta, docs = read_text_book(d)
        self.assertEqual([doc.get_name() for doc in docs], ['第1話.txt', '第2話.txt', '第10話.txt'])
        self.assertEqual(meta['title'], 'novel')
        self.assertTrue(str(meta['cover']).endswith('cover.png'))

    def test_dry_run_of_japanese_markdown(self):
        f = self.tmp / 'story.md'
        f.write_text('---\ntitle: 星の旅人\nauthor: 山田太郎\n---\n# 第1話　出会い\n\n雨が降っていた。' + '彼は傘を持っていなかった。' * 12 +
                     '\n\n# 第2話\n\n「失敗だ」と彼は言った。' + '朝が来た。' * 30 + '\n', encoding='utf-8')
        res = core.main(f, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True))
        chapters = res['chapters']
        self.assertEqual([c.title for c in chapters], ['第1話 出会い', '第2話'])
        self.assertEqual(chapters[0].segments[0].text, '星の旅人。')
        self.assertTrue(all(len(s.text) <= 110 for c in chapters for s in c.segments))


if __name__ == '__main__':
    unittest.main()
