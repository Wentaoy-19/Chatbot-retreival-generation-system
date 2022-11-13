import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def re_ranker(user_question = "", response_list = []):
    rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    features = rerank_msmarco_tokenizer([user_question] * len(response_list), response_list,  padding=True, truncation=True, return_tensors="pt")

    rerank_msmarco_model.eval()
    with torch.no_grad():
        scores = rerank_msmarco_model(**features).logits
    return scores
