import json
import os
import time
from openai import OpenAI

# API Key 설정 (환경 변수 또는 직접 입력)
# os.environ["OPENAI_API_KEY"] = "..."

BATCH_INFO_FILE = "batch_info.json"
BATCH_INPUT_FILE = "batch_input.jsonl"

def load_system_prompt():
    try:
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "You are an expert in {DOMAIN}. Output JSON only."

def create_batch_file(fake_data, system_template):
    """Batch API용 JSONL 파일 생성"""
    tasks = []
    
    for domain, details in fake_data.items():
        categories = details.get("Product Category", [])
        for category in categories:
            # Custom ID에 도메인과 카테고리 정보를 인코딩 (구분자 :: 사용)
            custom_id = f"{domain}::{category}"
            
            prompt_content = system_template.format(
                DOMAIN=domain,
                CATEGORY=category
            )
            
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o", # Batch API는 gpt-4o, gpt-4o-mini 등 지원
                    "messages": [
                        {"role": "system", "content": "You are a JSON generator. Output only valid JSON."},
                        {"role": "user", "content": prompt_content}
                    ],
                    "temperature": 0.7
                }
            }
            tasks.append(task)
            
    with open(BATCH_INPUT_FILE, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
            
    print(f"✅ Created batch input file with {len(tasks)} tasks: {BATCH_INPUT_FILE}")
    return len(tasks)

def submit_batch(client):
    """Batch 파일 업로드 및 작업 생성"""
    # 1. 파일 업로드
    batch_input_file = client.files.create(
        file=open(BATCH_INPUT_FILE, "rb"),
        purpose="batch"
    )
    print(f"⬆️  Uploaded file ID: {batch_input_file.id}")

    # 2. Batch 생성
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h" # 현재는 24h만 지원 (50% 할인)
    )
    
    print(f"🚀 Batch job created! ID: {batch_job.id}")
    print("   (It may take up to 24 hours, but usually faster for small batches)")
    
    # 정보 저장
    info = {
        "batch_id": batch_job.id,
        "file_id": batch_input_file.id,
        "status": "submitted",
        "created_at": time.time()
    }
    with open(BATCH_INFO_FILE, 'w') as f:
        json.dump(info, f, indent=4)
        
    return batch_job.id

def check_and_retrieve_results(client):
    """Batch 상태 확인 및 결과 다운로드"""
    with open(BATCH_INFO_FILE, 'r') as f:
        info = json.load(f)
        
    batch_id = info['batch_id']
    batch_job = client.batches.retrieve(batch_id)
    
    print(f"📊 Batch Status: {batch_job.status}")
    
    if batch_job.status == 'completed':
        print("⬇️  Downloading results...")
        result_file_id = batch_job.output_file_id
        
        content = client.files.content(result_file_id).text
        
        # 결과 처리 및 저장
        save_results(content)
        
        # 완료 표시 (파일 삭제 또는 상태 업데이트)
        print("✅ All files saved successfully!")
        os.remove(BATCH_INFO_FILE) # 작업 완료 후 정보 파일 삭제
        
    elif batch_job.status in ['failed', 'expired', 'cancelled']:
        print(f"❌ Batch failed: {batch_job.errors}")
    else:
        print("⏳ Batch is still processing. Please try again later.")

def save_results(result_content):
    """결과 JSONL 파싱 및 파일 저장"""
    base_spec_dir = "Base_Specs"
    
    for line in result_content.strip().split('\n'):
        if not line: continue
        
        res = json.loads(line)
        custom_id = res['custom_id']
        domain, category = custom_id.split('::')
        
        # 응답 내용 추출
        response_body = res['response']['body']
        if 'choices' in response_body:
            content = response_body['choices'][0]['message']['content']
            clean_json = content.replace("```json", "").replace("```", "").strip()
            
            try:
                spec_data = json.loads(clean_json)
                
                # 디렉토리 생성 및 저장
                domain_dir = os.path.join(base_spec_dir, domain)
                os.makedirs(domain_dir, exist_ok=True)
                
                filename = f"{category.replace(' ', '_')}_base_spec.json"
                filepath = os.path.join(domain_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(spec_data, f, indent=4)
                # print(f"   -> Saved {filepath}")
                
            except json.JSONDecodeError:
                print(f"⚠️ JSON Decode Error for {custom_id}")
        else:
            print(f"⚠️ Error in response for {custom_id}")

def main():
    client = OpenAI()
    
    # 이미 진행 중인 배치가 있는지 확인
    if os.path.exists(BATCH_INFO_FILE):
        print("🔄 Found existing batch job info.")
        check_and_retrieve_results(client)
    else:
        print("🆕 Starting new batch process...")
        # 1. 데이터 로드
        with open('Fake_data.json', 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
            
        system_template = load_system_prompt()
        
        # 2. 배치 파일 생성
        count = create_batch_file(fake_data, system_template)
        
        if count > 0:
            # 3. 배치 제출
            submit_batch(client)
        else:
            print("No tasks to process.")

if __name__ == "__main__":
    main()