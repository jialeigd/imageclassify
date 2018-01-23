import json


def write_data_to_json(file_name, dic):
    js_obj = json.dumps(dic)
    with open(file_name, 'w') as json_file:
        json_file.write(js_obj)

def read_data_from_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        return data
