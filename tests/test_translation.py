"""
번역 사전(src/api/ko_en_dict.json)과 사전 → opus-mt 순서 테스트
"""
from src.api.translation import DictTranslator, QueryTranslator


d = DictTranslator()


class TestDictTranslator:
    def test_single_word(self):
        # opus-mt가 "Indulgent"로 옮기던 단어
        assert d.translate("시집") == "poetry collection"

    def test_multiword_unit_before_single_words(self):
        # "비 오는 날"은 비(rain)+오는+날이 아니라 한 단위다
        assert d.translate("비 오는 날 재즈") == "rainy day jazz"

    def test_particle_is_stripped(self):
        assert d.translate("바다의 고양이") == "sea cats"

    def test_short_stem_does_not_strip_particle(self):
        # "시가"를 시(poetry)+가로 읽지 않는다 — 남는 말이 한 글자면 떼지 않는다
        assert d.translate("시가") is None

    def test_filler_is_dropped(self):
        assert d.translate("비 오는 날 듣기 좋은 노래") == "rainy day songs"

    def test_numbers_and_latin_pass_through(self):
        assert d.translate("재즈 3") == "jazz 3"
        assert d.translate("BTS 발라드") == "BTS ballad"

    def test_unknown_word_falls_back(self):
        # 하나라도 사전에 없으면 None — 부분 번역을 내지 않는다
        assert d.translate("재즈 냉장고") is None
        assert d.translate("인간이란 무엇인가") is None


class TestQueryTranslator:
    def test_dictionary_hit_does_not_load_mt(self):
        t = QueryTranslator()
        assert t.translate(["로파이", "lofi"]) == ["lo-fi", "lofi"]
        assert t._model is None
