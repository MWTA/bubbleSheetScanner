import os
import random
import string
import json
import sys
import argparse

PAPERID_LENGTH = 3

def append_zeros(integer, str_len):
    if integer >= 10 ** str_len:
        print 'Attempted to encode %d in %d digits' % (integer, str_len)
        raise Exception('Integer overflow in encoding')
    for i in range(1, str_len):
        if integer < 10 ** (i):
            res = '0' * (str_len - i) + str(integer)
            break
        else:
            res = str(integer)
    return res

def gen_papers(test_id, test_name, num_papers, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pins = {}
    for i in range(num_papers):
        paper_id = append_zeros(i, PAPERID_LENGTH)
        pin = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        path = os.path.join(out_path, 'paper_test%s_paper%s.pdf' % (test_id, paper_id))
        gen_bubble_sheet(test_id, paper_id, pin, test_name, path)
        pins[paper_id] = pin
    index_file = os.path.join(out_path, 'pins.json')
    fh = open(index_file, 'w')
    fh.write(json.dumps(pins))
    fh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_id", help="The five-digit ASCII ID of the test")
    parser.add_argument("test_name", help="The human-readable title of the test")
    parser.add_argument("num_papers", help="Number of papers to generate", type=int)
    parser.add_argument("out_path", help="Path of output files")
    args = parser.parse_args()
    gen_papers(args.test_id, args.test_name, args.num_papers, args.out_path)
