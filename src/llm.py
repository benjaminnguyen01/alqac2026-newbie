import random

import ollama

def predict(model: str, case_query: str, case_text: str, law_texts: list) -> str:
    combined_law = "\n\n".join([f"DL ({law['law_id']} - ID {law['aid']}):\n{law['text']}"  for law in law_texts])

    max_fact_length = 8192
    truncated_case_text = case_text[:max_fact_length] + ("..." if len(case_text) > max_fact_length else "")

    prompt = (f"No yapping. Bạn là luật sư chuyên về luật Việt Nam. Dùng thông tin truy vấn, tình tiết vụ án và các điều luật dưới đây. "
              f"TRUY VẤN: {case_query}. TÌNH TIẾT: {truncated_case_text}. ĐIỀU LUẬT: {combined_law}. "
              f"Dự đoán kết quả vụ án, bắt buộc chỉ trả về một trong 4 nhãn sau: A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN, B_WIN. "
              f"Nhãn dự đoán là?")
    answer = ollama.chat(model = model, messages=[
        {"role": "system", "content": "Là một hệ thống pháp luật, hãy dự đoán chính xác 1 trong 4 nhãn."},
        {"role": "user", "content": prompt}
    ])
    output = answer["message"]["content"].strip()

    valid_labels = ["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]
    for label in valid_labels:
        if label in output:
            return label
    return random.choice(valid_labels)