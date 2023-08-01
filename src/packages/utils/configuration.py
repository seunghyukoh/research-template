import json
import pprint

__all__ = ["ConfLoader"]


class ConfLoader:
    """
    Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """

    class DictWithAttributeAccess(dict):
        """
        This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

        def __getstate__(self):
            return self.__dict__.copy()

        def __setstate__(self, state):
            self.__dict__.update(state)

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()

    def __load_conf(self):
        with open(self.conf_name, "r") as conf:
            opt = json.load(
                conf, object_hook=lambda dict: self.DictWithAttributeAccess(dict)
            )

            # Print configuration dictionary pretty
            print("")
            print("=" * 50 + " Configuration " + "=" * 50)
            pp = pprint.PrettyPrinter(compact=True)
            pp.pprint(opt)
            print("=" * 120)

        return opt

    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)

        return opt
