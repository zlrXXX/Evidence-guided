import re

MODEL = "gpt-5.6-terra"
CRITERIA = ['general_1_understanding', 'general_2_interpersonal_effectiveness', 'general_3_collaboration', 'CBT_1_guided_discovery', 'CBT_2_focus', 'CBT_3_strategy']

def render_transcript(turns):
    if not isinstance(turns, list) or not turns:
        raise ValueError("Dialogue must be a nonempty JSON array")
    rendered = []
    expected = "counsellor"
    for index, turn in enumerate(turns):
        if set(turn) != {"speaker", "utterance"}:
            raise ValueError(f"Unexpected transcript fields at turn {index}")
        speaker = turn["speaker"]
        utterance = turn["utterance"]
        if speaker != expected or not isinstance(utterance, str) or not utterance.strip():
            raise ValueError(f"Invalid speaker order or text at turn {index}")
        rendered.append(("咨询师：" if speaker == "counsellor" else "来访者：") + utterance.strip())
        expected = "patient" if expected == "counsellor" else "counsellor"
    return "\n".join(rendered)

def parse_result(row):
    if row.get('error') or row.get('response', {}).get('status_code') != 200:
        raise ValueError('request_failed')
    body = row['response']['body']
    choice = body['choices'][0]
    if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
        raise ValueError('incomplete_or_refused')
    content = choice['message']['content']
    match = re.fullmatch(r'\s*([0-6])\s*(?:分)?\s*[,，]\s*(\S[\s\S]*)', content)
    if not match:
        raise ValueError('invalid_score_explanation_format')
    return int(match.group(1)), match.group(2).strip(), body
