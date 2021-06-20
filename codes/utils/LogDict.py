import inspect


class LogDict:
    def __init__(self, name):
        self.d = {}
        self.name = name

    def add(self, name, val=None):
        if val:
            self.d[name] = val
            return

        frame = inspect.stack()[1][0]
        while name not in frame.f_locals:
            frame = frame.f_back
            if frame is None:
                return None
        var = frame.f_locals[name]

        self.d[name] = var

    @staticmethod
    def to_string(val):
        if isinstance(val, float):
            return "{:.2E}".format(val)
        return str(val)

if __name__ == "__main__":
    l = LogDict('t')

    test = "1"
    test2 = "2"

    l.add('test')
    l.add('test2')

    print(l)
