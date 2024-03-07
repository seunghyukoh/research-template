import os


def directory_setter(path="./results", make_dir=False):
    if not os.path.exists(path) and make_dir:
        os.makedirs(path)  # make dir if not exist
        print("directory %s is created" % path)

    if not os.path.isdir(path):
        raise NotADirectoryError(
            "%s is not valid. set make_dir=True to make dir." % path
        )
