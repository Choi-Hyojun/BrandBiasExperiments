from huggingface_hub import hf_hub_download, list_repo_files, HfFileSystem
import pandas as pd
import json
import os

# 2023 데이터셋의 정확한 카테고리명 매핑 (공식 레포 기준)
CATEGORY_MAP = {
    # --- Group A: Spec/Fact Driven ---
    "Cell_Phones": "Cell_Phones_and_Accessories",
    "Automotive": "Automotive",
    "Tools": "Tools_and_Home_Improvement",

    # --- Group B: Review/Subjective Driven ---
    "Fashion": "Clothing_Shoes_and_Jewelry",
    "Beauty": "All_Beauty",  # 이름이 'All_Beauty'로 변경됨
    "Movies": "Movies_and_TV"
}

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

def check_file_exists(filename):
    """레포지토리에 파일이 실제로 존재하는지 미리 확인"""
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    if filename in all_files:
        return True
    return False

def get_amazon_2023_data(category_key, num_samples=5000):
    hf_category = CATEGORY_MAP[category_key]
    print(f"🔍 Checking files for category: {hf_category}...")

    # 1. 리뷰 파일 경로 찾기
    # 예상 경로들 (구조가 가끔 바뀌어서 여러개 시도)
    possible_review_paths = [
        f"raw/review_categories/{hf_category}.jsonl",
        f"raw/review_categories/{hf_category}.jsonl.gz", 
        f"raw/review/{hf_category}.jsonl"
    ]
    
    review_file = None
    all_repo_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    for path in possible_review_paths:
        if path in all_repo_files:
            review_file = path
            break
            
    if not review_file:
        print(f"❌ Cannot find review file for {hf_category}")
        print("Available folders in repo:", {f.split('/')[0] for f in all_repo_files})
        return None, None

    # 2. 메타 파일 경로 찾기
    possible_meta_paths = [
        f"raw/meta_categories/meta_{hf_category}.jsonl",
        f"raw/meta_categories/{hf_category}.jsonl",
        f"raw/meta/meta_{hf_category}.jsonl"
    ]
    
    meta_file = None
    for path in possible_meta_paths:
        if path in all_repo_files:
            meta_file = path
            break

    # 3. 데이터 스트리밍 (다운로드 없이 읽기)
    fs = HfFileSystem()
    
    def read_remote_jsonl(file_path, limit):
        if not file_path: return pd.DataFrame()
        
        data = []
        full_path = f"datasets/{REPO_ID}/{file_path}"
        print(f"📖 Streaming first {limit} rows from {full_path}...")
        
        try:
            with fs.open(full_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit: break
                    try:
                        data.append(json.loads(line))
                    except:
                        continue
        except Exception as e:
            print(f"⚠️ Error reading {full_path}: {e}")
            
        return pd.DataFrame(data)

    df_reviews = read_remote_jsonl(review_file, num_samples)
    df_meta = read_remote_jsonl(meta_file, num_samples)

    return df_reviews, df_meta

# --- 실행 ---
if __name__ == "__main__":
    base_dir = "data_2023"
    os.makedirs(base_dir, exist_ok=True)

    for category_key in CATEGORY_MAP.keys():
        print(f"\n🚀 Processing category: {category_key}")
        
        # 카테고리별 디렉토리 생성
        category_dir = os.path.join(base_dir, category_key)
        os.makedirs(category_dir, exist_ok=True)
        
        # 데이터 다운로드 (샘플 수 조절 가능)
        df_r, df_m = get_amazon_2023_data(category_key, num_samples=1000)
        
        if df_r is not None:
            print(f"✅ Success! Loaded {len(df_r)} reviews for {category_key}.")
            # print(df_r[['rating', 'title', 'text']].head(2))
            
            # 저장
            review_path = os.path.join(category_dir, f"{category_key}_reviews.csv")
            meta_path = os.path.join(category_dir, f"{category_key}_meta.csv")
            
            df_r.to_csv(review_path, index=False)
            print(f"   -> Saved reviews to {review_path}")
            
            if df_m is not None and not df_m.empty:
                df_m.to_csv(meta_path, index=False)
                print(f"   -> Saved meta to {meta_path}")
        else:
            print(f"❌ Failed to load data for {category_key}")