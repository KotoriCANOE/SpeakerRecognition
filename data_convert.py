import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from utils import eprint, listdir_files

def subfunc(ipath, opath, d, ffmpeg='ffmpeg', iext='.m4a', oext='.wav'):
    ifiles = listdir_files(d, filter_ext=[iext])
    for ifile in ifiles:
        ofile = ifile.replace(ipath, opath, 1)
        ofile = os.path.splitext(ofile)[0] + oext
        save_path = os.path.split(ofile)[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif os.path.exists(ofile):
            continue
        cmdline = '{} -y -i {} -vn -c:a pcm_s16le {}'.format(
            ffmpeg, ifile, ofile)
        std_redirect = subprocess.PIPE
        subprocess.run(cmdline.split(), stdout=std_redirect, stderr=std_redirect)

def convert(ipath, opath, threads=4):
    dataset_ids = os.listdir(ipath)
    regex = re.compile(r'^id(\d+)$')
    ids = [re.findall(regex, i) for i in dataset_ids]
    dataset_ids = [(int(i[0]), os.path.join(ipath, d)) for i, d
        in zip(ids, dataset_ids) if i and i[0].isnumeric()]
    if threads == 1:
        for i, d in dataset_ids:
            subfunc(ipath, opath, d)
    else:
        with ThreadPoolExecutor(threads) as executor:
            for i, d in dataset_ids:
                executor.submit(subfunc, ipath, opath, d)

def main(argv):
    import argparse
    argp = argparse.ArgumentParser()
    argp.add_argument('input')
    argp.add_argument('output')
    argp.add_argument('-t', '--threads', type=int, default=4)
    args = argp.parse_args(argv)
    convert(args.input, args.output, args.threads)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
