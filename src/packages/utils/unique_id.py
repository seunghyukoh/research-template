import hashlib


def sort_dict_by_key(data: dict):
    return dict(sorted(data.items(), key=lambda item: item[0]))


def generate_unique_id(data: dict):
    # return hashlib.md5(str(data).encode()).hexdigest()
    return hashlib.md5(str(sort_dict_by_key(data)).encode()).hexdigest()
