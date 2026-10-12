"""
검색어 원문에서 도메인 명시를 읽는 규칙(src/api/domain_intent.py) 테스트
"""
import pytest

from src.api.domain_intent import detect_domain


CASES = {
    # 도메인 명사가 있으면 제한한다
    "잔잔한 영화 3": "movie",
    "비 오는 날 듣기 좋은 노래": "music",
    "비 오는 날 듣기 좋은 음악": "music",
    "시집": "book",
    "그림책": "book",
    "신곡": "music",
    # 합성어도 그 명사로 끝나면 단서다
    "추리소설 로맨스": "book",
    "발라드 에세이": "book",
    "공포영화": "movie",
    "연주곡": "music",
    # 장르·상황만 있으면 all (10/12 결정: 한 도메인뿐인 장르도 제한하지 않는다)
    "발라드": "all",
    "힙합": "all",
    "재즈": "all",
    "다큐멘터리": "all",
    "산책할 때": "all",
    # 그 명사로 끝나지만 작품 종류가 아닌 말
    "산책": "all",
    "정책": "all",
    "계곡": "all",
    "희곡": "all",
    "리플리": "all",
    # 비유로 쓰인 쪽은 버리고 핵심 명사를 고른다
    "영화 같은 노래": "music",
    "영화같은 노래": "music",
    "책처럼 읽히는 앨범": "music",
    "영화 OST": "music",
    "책 읽을 때 듣는 음악": "music",
    "노래를 들으며 읽는 책": "book",
    "영화를 보고 싶어": "movie",
    # 여러 도메인을 함께 원하면 all
    "영화나 책": "all",
    "영화와 책": "all",
    "책과 영화 추천": "all",
    # 영어
    "calm movies": "movie",
    "music for reading books": "music",
    "songs from movies": "music",
    "books about music": "book",
    "a novel about jazz": "book",
    "movie soundtrack": "music",
    "movie-like songs": "music",
    "movies or books": "all",
    "jazz": "all",
    "rainy day": "all",
    "jazz 영화": "movie",
    "": "all",
}


@pytest.mark.parametrize("query,expected", list(CASES.items()))
def test_detect_domain(query, expected):
    assert detect_domain(query) == expected
