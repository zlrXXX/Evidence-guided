import copy
import json
ENGAGEMENT_CUES = {'REJECTING_DISENGAGED': 'The patient is currently strongly guarded or rejecting; respond with limited engagement unless the visible dialogue itself supports a small shift.', 'AMBIVALENT': 'The patient is currently mixed or uncertain; express both hesitation and some willingness where supported by the visible dialogue.', 'COOPERATIVE': 'The patient is currently willing to engage with the counsellor while retaining realistic uncertainty or difficulty.', 'COLLABORATIVE_EXPLORATION': 'The patient is currently exploring the issue collaboratively while still speaking from their own experience and without assumed independent mastery.'}
COUNSELLOR_CUES = {'REJECTING_DISENGAGED': 'The patient is guarded, doubtful, and only minimally willing to engage; do not force agreement.', 'AMBIVALENT': 'The patient is willing to engage in part but remains uncertain and may hold reservations.', 'COOPERATIVE': 'The patient is willing to work with the counsellor while remaining realistic rather than automatically agreeing.', 'COLLABORATIVE_EXPLORATION': 'The patient is actively exploring and contributing while remaining capable of uncertainty or correction.'}
PATIENT_KEYS = {'patient_profile', 'symptom_level', 'counselling_progress', 'current_resistance_state', 'dialogue_history', 'current_counsellor_utterance'}

def history_turns(history):
    turns = []
    for exchange in history:
        if set(exchange) != {'counsellor', 'patient'}:
            raise ValueError('Invalid causal dialogue exchange')
        for role in ('counsellor', 'patient'):
            turns.append({'role': role, 'utterance': exchange[role]})
    return turns

def counsellor_context(visible, response=False, rloo=False):
    profile = copy.deepcopy(visible['patient_profile'])
    profile['symptom_level'] = copy.deepcopy(visible['symptom_level'])
    shown = {'patient_profile': profile, 'supported_session_information': copy.deepcopy(visible['counselling_progress']), 'dialogue_history': history_turns(visible['dialogue_history']), 'active_cbt_rag': copy.deepcopy(visible['active_cbt_rag']), 'cbt_progress_by_step': copy.deepcopy(visible['cbt_progress_by_step']), 'current_cbt_progress_state': visible['current_cbt_progress_state'], 'completed_technique_ids': copy.deepcopy(visible['completed_technique_ids']), 'verified_prior_patient_evidence': copy.deepcopy(visible['current_patient_evidence_span'])}
    if rloo:
        shown.pop('current_cbt_progress_state')
        active = visible['active_cbt_rag']
        shown['active_step_progress_state_before_decision'] = visible['cbt_progress_by_step'].get(f'{active['technique_id']}:{active['current_step']['step_id']}', 'NO_DEMONSTRATION') if active else 'NO_ACTIVE_TECHNIQUE'
    if response:
        decision = visible['selected_decision']
        shown['natural_engagement_cue'] = COUNSELLOR_CUES[visible['current_resistance_state']]
        shown['selected_controller_decision'] = copy.deepcopy(decision)
        shown['selected_official_cbt_step'] = copy.deepcopy(visible['selected_cbt_rag'])
        shown['response_phase'] = 'SWITCH_INTRODUCTION' if rloo and decision['action'] == 'SWITCH_TECHNIQUE' else 'END_PROPOSAL' if decision['session_control'] == 'END_SESSION' else 'IN_SESSION_RESPONSE'
    else:
        shown['current_provisional_resistance_state'] = visible['current_resistance_state']
        shown['cbt_rag_options'] = copy.deepcopy(visible['cbt_rag_options'])
    return shown

def patient_context(visible, rloo=False):
    keys = set(visible)
    if not PATIENT_KEYS <= keys or not keys <= PATIENT_KEYS | {'response_phase'}:
        raise ValueError('Patient-visible context violates the causal boundary')
    shown = copy.deepcopy(visible)
    shown.setdefault('response_phase', 'IN_SESSION_RESPONSE')
    if rloo:
        shown['provisional_engagement_cue'] = ENGAGEMENT_CUES[shown.pop('current_resistance_state')]
    else:
        shown['patient_profile']['symptom_level'] = shown.pop('symptom_level')
    return shown

def patient_system(prompts, phase):
    if phase == 'PATIENT_END_ACCEPTANCE':
        return prompts['patient_end_acceptance_system']
    if phase == 'PATIENT_END_CONTINUE':
        return prompts['patient_end_continue_system']
    if phase in {'IN_SESSION_RESPONSE', 'POST_CLOSING_ACKNOWLEDGEMENT'}:
        return prompts['patient_system']
    raise ValueError(f'Unknown patient response phase: {phase}')

def action_candidates(visible, rag, validate=None):
    from action_space import action_text, legal_action_candidates
    candidates = legal_action_candidates(counsellor_context(visible, rloo=True), rag)
    if not candidates:
        raise ValueError('No legal action for this causal context')
    return [action_text(candidate) for candidate in candidates]
