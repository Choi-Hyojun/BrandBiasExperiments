# 모델 이름은 사용하시는 체크포인트로 바꾸세요 (예: "skt/kogpt2-base-v2" 등)
model_name = "gpt2"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@torch.no_grad()
def seq_logprob(model, tokenizer, prefix: str, phrase: str):
    # prefix: 문맥(빈문자열 가능), phrase: 비교할 단어/문구
    # 반환: (log_prob_sum, prob) -- log prob in natural log
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    phrase_ids = tokenizer(phrase, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    # context 시작은 prefix (없으면 빈 텐서)
    if prefix_ids.size(1) == 0:
        context = torch.empty((1,0), dtype=torch.long, device=device)
    else:
        context = prefix_ids.clone()
    total_logprob = 0.0
    for tok in phrase_ids[0]:
        # 모델에 현재 context를 넣고 다음 토큰 분포 얻기
        if context.size(1) == 0:
            # 일부 토크나이저/모델은 empty context 불가 -> 대신 feed first token as context step-by-step
            # 여기서는 빈 context 허용 안되는 모델이면 tokenizer에 BOS 필요 여부 확인하세요.
            input_ids = tok.unsqueeze(0).unsqueeze(0)  # fallback (not ideal)
        else:
            input_ids = context
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
        last_logits = logits[0, -1, :]  # next-token 분포
        log_probs = torch.log_softmax(last_logits, dim=-1)
        tok_id = int(tok)
        token_logprob = float(log_probs[tok_id].cpu().item())
        total_logprob += token_logprob
        # append the true token to context for next step
        tok_tensor = tok.unsqueeze(0).unsqueeze(0)  # shape (1,1)
        context = torch.cat([context, tok_tensor], dim=1) if context.size(1) > 0 else tok_tensor
    # 확률 (underflow 주의): exp(logprob)
    prob = math.exp(total_logprob) if total_logprob > -1000 else 0.0
    return total_logprob, prob

# 예시 비교 (한국어 예시)
prefix = ""  # 문맥이 있다면 여기에 넣으세요
a = "삼성 갤럭시S 25"
b = "애플 아이폰 24"

logpa, pa = seq_logprob(model, tokenizer, prefix, a)
logpb, pb = seq_logprob(model, tokenizer, prefix, b)

print("A:", a, "logprob:", logpa, "prob:", pa)
print("B:", b, "logprob:", logpb, "prob:", pb)
if pa>0 and pb>0:
    print("A/B 확률 비율:", pa/pb, "A 우위(로그):", logpa-logpb)
else:
    print("확률이 매우 작아 비율 계산 불가 (로그 차이):", logpa-logpb)
