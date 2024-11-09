import json

def write_json_line(data, fout):
    json.dump(data, fout, separators=(',', ':'))
    print(file=fout)

def load_jsonl_file(fin):
    for line in fin:
        yield json.loads(line)
