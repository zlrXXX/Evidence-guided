import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from ctrs import CRITERIA, load, validate
from transport import check, digest, save

BOOTSTRAPS = 10_000
PERMUTATIONS = 100_000
SEED = 20261002


def seed_for(name):
    return SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def bootstrap(values, seed):
    rng = np.random.default_rng(seed)
    draws = np.concatenate([values[rng.integers(0, len(values), (500, len(values)))].mean(axis=1)
                            for _ in range(BOOTSTRAPS // 500)])
    return np.quantile(draws, [.025, .975]).tolist()


def read_group(spec, base, allow_incomplete):
    run = (base / spec['run']).resolve()
    manifest = validate(run)
    cohort = next(c for c in load(run / 'cohorts.json') if c['cohort'] == spec['cohort'])
    path = run / 'results/scores.json'
    records = [r for r in load(path) if r['cohort'] == spec['cohort']]
    source = {r['session_id']: r for r in load(run / 'source-index.json') if r['cohort'] == spec['cohort']}
    expected_profiles = {r['profile_id'] for r in source.values()}
    by_profile = collections.defaultdict(dict)
    for r in records:
        check(r['session_id'] in source and r['profile_id'] == source[r['session_id']]['profile_id'], 'Profile mismatch')
        check(r['source_sha256'] == source[r['session_id']]['source_sha256'] and
              r['transcript_sha256'] == source[r['session_id']]['transcript_sha256'], 'Score source mismatch')
        check(type(r['score']) is int and 0 <= r['score'] <= 6 and r['criterion'] in CRITERIA, 'Invalid score')
        check(r['criterion'] not in by_profile[r['profile_id']], 'Duplicate score')
        by_profile[r['profile_id']][r['criterion']] = r['score']
    complete = sorted(pid for pid in expected_profiles if set(by_profile[pid]) == set(CRITERIA))
    incomplete = sorted(expected_profiles - set(complete))
    check(allow_incomplete or not incomplete, 'Incomplete group: ' + spec['name'] + '; collect remaining scores or explicitly use --allow-incomplete')
    check(complete, 'No complete profiles')
    values = np.array([[by_profile[pid][c] for c in CRITERIA] for pid in complete], dtype=np.int64)
    row = {'name': spec['name'], 'cohort': spec['cohort'], 'dataset': cohort['dataset'],
           'backbone': cohort['backbone'], 'language': manifest['language'], 'n': len(complete),
           'expected_n': len(expected_profiles), 'incomplete_profiles': incomplete,
           'profile_ids': complete, 'criterion_means': values.mean(axis=0).tolist(),
           'criterion_sample_sd': values.std(axis=0, ddof=1).tolist() if len(complete) > 1 else None,
           'overall_mean': float(values.mean()),
           'overall_ci95': bootstrap(values.mean(axis=1), spec.get('bootstrap_seed', seed_for(spec['name']))),
           'score_file_sha256': digest(path.read_bytes()), 'bootstrap_seed': spec.get('bootstrap_seed', seed_for(spec['name']))}
    return row, dict(zip(complete, values.sum(axis=1)))


def compare(reference, variant, reference_totals, variant_totals, family, permutation_seed=None):
    check((reference['dataset'], reference['backbone'], reference['language']) ==
          (variant['dataset'], variant['backbone'], variant['language']), 'Comparison dataset/backbone/language differs')
    ids = sorted(reference_totals.keys() & variant_totals.keys())
    check(ids, 'No matched profiles')
    differences = np.array([reference_totals[pid] - variant_totals[pid] for pid in ids], dtype=np.int64)
    observed = abs(int(differences.sum()))
    nonzero = differences[differences != 0]
    seed = permutation_seed if permutation_seed is not None else seed_for(reference['name'] + '|' + variant['name'])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(PERMUTATIONS // 1000):
        signs = rng.integers(0, 2, (1000, len(nonzero)), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(abs(signs @ nonzero) >= observed))
    return {'reference': reference['name'], 'variant': variant['name'], 'family': family,
            'paired_n': len(ids), 'matched_profile_ids': ids,
            'reference_only_profiles': sorted(reference_totals.keys() - variant_totals.keys()),
            'variant_only_profiles': sorted(variant_totals.keys() - reference_totals.keys()),
            'reference_minus_variant_mean': float(differences.mean() / 6),
            'difference_ci95': bootstrap(differences / 6, seed + 1),
            'p_raw': (extreme + 1) / (PERMUTATIONS + 1), 'extreme_count': extreme, 'seed': seed}


def main():
    parser = argparse.ArgumentParser(description='Profile bootstrap and paired CTRS permutation tests.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    config = load(args.config)
    check(len({g['name'] for g in config['groups']}) == len(config['groups']), 'Duplicate group')
    groups = {g['name']: read_group(g, args.config.resolve().parent, args.allow_incomplete) for g in config['groups']}
    tests = []
    for comparison in config['comparisons']:
        ref, ref_values = groups[comparison['reference']]
        var, var_values = groups[comparison['variant']]
        tests.append(compare(ref, var, ref_values, var_values, comparison['family'], comparison.get('permutation_seed')))
    check(len({(t['reference'], t['variant'], t['family']) for t in tests}) == len(tests), 'Duplicate comparison')
    check(set(config['families']) == {t['family'] for t in tests}, 'Comparison families mismatch')
    for family, expected in config['families'].items():
        family_tests = [t for t in tests if t['family'] == family]
        check(len(family_tests) == expected, 'Incomplete Holm family: ' + family)
        running = 0.0
        for rank, test in enumerate(sorted(family_tests, key=lambda x: x['p_raw'])):
            running = max(running, min(1.0, (len(family_tests) - rank) * test['p_raw']))
            test['p_holm'] = running
    save(args.output, {'bootstrap_resamples': BOOTSTRAPS, 'permutations': PERMUTATIONS,
                      'p_value_estimator': '(extreme + 1) / (permutations + 1)',
                      'confidence_interval': '95% percentile profile bootstrap',
                      'test': 'two-sided paired sign-flip test on six-criterion profile sums',
                      'families': config['families'], 'rows': [g[0] for g in groups.values()], 'tests': tests})
    print(json.dumps({'groups': len(groups), 'comparisons': len(tests), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
