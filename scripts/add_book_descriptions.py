"""
Data preprocessing script for adding book descriptions from Google Books API.
Fetches book descriptions by ISBN and adds them to CSV dataset.
"""
import pandas as pd
import requests
import time


def get_book_description(isbn):
    """
    Google Books API를 사용하여 ISBN으로 책의 description(소개글)과 category(장르)를 가져옵니다.
    
    Args:
        isbn: ISBN 번호 (10 또는 13자리)
        
    Returns:
        tuple: (description, category) - 책의 설명과 장르
    """
    # ISBN 번호에서 하이픈(-)이나 공백을 완전히 제거하여 숫자만 남깁니다.
    isbn_clean = str(isbn).strip().replace('-', '').replace(' ', '')
    
    # 10자리 또는 13자리 숫자가 아니면 패스 (유효하지 않은 데이터 처리)
    if not isbn_clean or len(isbn_clean) not in [10, 13]:
        return "", "Unknown"
        
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn_clean}"
    
    try:
        # API 호출 (타임아웃 10초 설정)
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            # 검색 결과가 존재하는지 확인
            if "items" in res_json and len(res_json["items"]) > 0:
                volume_info = res_json["items"][0].get("volumeInfo", {})
                # description 필드가 있으면 가져오고, 없으면 빈 문자열 반환
                description = volume_info.get("description", "")
                
                # 장르 정보 가져오기 (리스트 형태이므로 첫 번째 요소를 가져옴)
                categories = volume_info.get("categories", ["Unknown"])
                category = categories[0] if categories else "Unknown"
                
                return description, category
    except Exception as e:
        print(f"\n[오류] ISBN {isbn} 검색 중 문제가 발생했습니다: {e}")
    
    return "", "Unknown"


def add_descriptions_to_csv(input_filename, output_filename):
    """
    CSV 파일의 각 행에 대해 ISBN으로 책 설명을 조회하여 추가합니다.
    
    Args:
        input_filename: 입력 CSV 파일 경로
        output_filename: 출력 CSV 파일 경로 (description 컬럼 추가됨)
    """
    print(f"'{input_filename}' 파일을 읽는 중...")
    df = pd.read_csv(input_filename)
    
    # CSV 내에 'isbn' 컬럼명이 대소문자가 다르거나 한글일 경우를 위한 보정 로직
    if 'isbn' not in df.columns:
        possible_isbn_cols = [col for col in df.columns if col.lower() in ['isbn', 'isbn코드', '도서번호']]
        if possible_isbn_cols:
            df.rename(columns={possible_isbn_cols[0]: 'isbn'}, inplace=True)
        else:
            print("에러: CSV 파일에 'isbn' 컬럼을 찾을 수 없습니다. 컬럼명을 확인해 주세요.")
            return

    descriptions = []
    categories = []
    total_rows = len(df)
    
    print("Google Books API를 통해 책 설명과 장르 정보를 수집하기 시작합니다...")
    
    for idx, row in df.iterrows():
        isbn = row['isbn']
        # 진행 상황 출력을 위해 타이틀 가져오기 (컬럼명이 다를 경우 예외처리)
        title = row.get('title', row.get('name', row.get('책이름', f"Row_{idx+1}")))
        
        print(f"[{idx + 1}/{total_rows}] {title} (ISBN: {isbn}) 데이터 요청 중...", end="")
        
        # API 호출하여 설명과 장르 가져오기
        desc, category = get_book_description(isbn)
        descriptions.append(desc)
        categories.append(category)
        
        if desc:
            print(f" -> 성공 (장르: {category})")
        else:
            print(" -> 실패 (데이터 없음)")
            
        # 안정적인 API 호출을 위한 미세한 딜레이 (초당 호출 제한 방지)
        time.sleep(0.3)
        
    # 데이터프레임에 새로운 컬럼 추가
    df['description'] = descriptions
    df['category'] = categories
    
    # 한국어 및 유니코드 깨짐 방지(엑셀 호환)를 위해 'utf-8-sig' 인코딩으로 저장
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n작업이 완료되었습니다! 결과 파일: '{output_filename}'")


if __name__ == "__main__":
    # ===================================================================
    # 1. 본인의 CSV 파일명을 입력하세요. (예: 'data/books_dataset.csv')
    # 2. 결과물로 저장할 파일명을 입력하세요.
    # ===================================================================
    INPUT_FILE = 'data/books_dataset.csv'  # 가지고 계신 파일명으로 바꾸세요.
    OUTPUT_FILE = 'data/books_dataset_with_description.csv'
    
    add_descriptions_to_csv(INPUT_FILE, OUTPUT_FILE)
