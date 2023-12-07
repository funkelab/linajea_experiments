NOTES ON MANUAL CURATION OF MOUSE TRACKS

After collecting MaMuT XMLs/csvs into one txt file (`tracks_uncorrected.txt`), the following
processing was done to clean up and curate the dataset:

- Remove all unattached nodes (nodes with no edges) - automatic identification with `../../scripts/check_ground_truth.py` and correction 
- Remove one edge with length >150 - automatic identification with `../../scripts/check_ground_truth.py` and manual correction
- Fix four edges that skipped over a frame - automatic identification with `../../scripts/check_ground_truth.py` and manual correction
- Remove duplicate tracks - automatic identification of candidates with `../../scripts/check_duplicate_gt_and_missing_edges.py`, manual identification of types of correction to make in MaMuT, automatic adjustment of tracks.txt with `remove_duplicate_tracks.py`
- Add in missing edges - automatic identification of candidates with `../../scripts/check_duplicate_gt_and_missing_edges.py`, manual identification of edges to add by inspection in MaMuT, automatic adjustment of tracks.txt with `add_missing_edges.py`
