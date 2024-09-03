import os
import sys
import argparse
import fnmatch
import subprocess

ProjectEXT = ['h', 'hpp', 'c', 'cpp']


def glob_files(dirs):
    files = []
    for directory in dirs:
        for root, _, filenames in os.walk(directory):
            for ext in ProjectEXT:
                for filename in fnmatch.filter(filenames, '*.' + ext):
                    files.append(os.path.join(root, filename))
    return files


if sys.platform == "linux":
    ClangBin = 'clang-format'
elif sys.platform == 'win32':
    ClangBin = 'clang-format.exe'


def clang_format_dir(args):
    files = glob_files(args.dirs)
    for file in files:
        cmds = [ClangBin, '-i', '--style=file', file]
        subprocess.run(cmds, check=True)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='Recursively clang-format')
    parser.add_argument('--dirs', nargs='+', help='paths to clang-format')
    args = parser.parse_args(argv[1:])
    if not args.dirs:
        sys.exit(-1)
    return args


if __name__ == '__main__':
    if len(sys.argv) == 1:
        args = parse_args(['', '--dirs', 'include', 'examples', 'tests'])
    else:
        args = parse_args()
    clang_format_dir(args)
