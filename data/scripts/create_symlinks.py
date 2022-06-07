import argparse
import os
import glob
import re


def create_symlinks(source_dir, target_dir, filename=None):
    assert os.path.isdir(source_dir)
    assert os.path.isdir(target_dir)

    if filename:
        pathname = os.path.join(source_dir, filename)
        source_files = [os.path.basename(p) for p in glob.glob(pathname)]
    else:
        source_files = [f for f in os.listdir(source_dir) if f.endswith('klb')]

    print(source_files)

    for source_file in source_files:
        target_name = "SPM%s_TM%s.klb"
        pattern = re.compile(".*SPM([0-9]*)_.*TM([0-9]*)_.*\\.klb")
        match = re.match(pattern, source_file)
        spm = match.group(1)
        tm = match.group(2)

        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, target_name % (spm, tm))
        print("Creating symlink from %s to %s" % (target_path, source_path))
        os.symlink(source_path, target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'source_dir',
            type=str,
            help='Path to directory containing klb files')
    parser.add_argument(
            'target_dir',
            type=str,
            help='Path to directory to create links in')
    parser.add_argument(
            '--limit_to',
            type=str,
            default=None,
            help='Filename format (unix style wildcards). Will only link to files with this format.')
    args = parser.parse_args()
    create_symlinks(args.source_dir, args.target_dir, args.limit_to)
