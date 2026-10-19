import csv
from download_data import write_movies


def test_unknown_genre_flag_is_skipped_and_western_preserved(tmp_path):
    # unknown=0, Action=1, all intermediate genre flags=0, Western=1.
    bits = ['0', '1'] + ['0'] * 16 + ['1']
    (tmp_path / 'u.item').write_text('|'.join(['1', 'Example', '01-Jan-1995', '', 'url'] + bits) + '\n')
    output = tmp_path / 'movies.csv'
    write_movies(str(tmp_path), str(output))
    with output.open() as f:
        row = next(csv.DictReader(f))
    assert row['genres'] == 'Action|Western'
