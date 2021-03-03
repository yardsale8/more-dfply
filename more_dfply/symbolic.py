from dfply.base import _context_args

class Intention(object):
    def __init__(self, function=lambda x: x, invert=False):
        self.function = function
        self.inverted = invert

    def evaluate(self, context):
        return self.function(context)

    def __getattr__(self, attribute):
        return Intention(lambda x: getattr(self.function(x), attribute),
                         invert=self.inverted)

    def __invert__(self):
        return Intention(self.function, invert=not self.inverted)

    def __call__(self, *args, **kwargs):
        return Intention(lambda x: self.function(x)(*_context_args(args)(x),
                                                    **_context_kwargs(kwargs)(x)),
                         invert=self.inverted)

    
_binary_reflections = {
    '__add__':'__radd__',
    '__and__':'__rand__',
    '__cmp__':'__rcmp__',
    '__div__':'__rdiv__',
    '__divmod__':'__rdivmod__',
    '__floordiv__':'__rfloordiv__',
    '__lshift__':'__rlshift__',
    '__mod__':'__rmod__',
    '__mul__':'__rmul__',
    '__or__':'__ror__',
    '__pow__':'__rpow__',
    '__rshift__':'__rrshift__',
    '__sub__':'__rsub__',
    '__truediv__':'__rtruediv__',
    '__xor__':'__rxor__',
    '__radd__':'__add__',
    '__rand__':'__and__',
    '__rcmp__':'__cmp__',
    '__rdiv__':'__div__',
    '__rdivmod__':'__divmod__',
    '__rfloordiv__':'__floordiv__',
    '__rlshift__':'__lshift__',
    '__rmod__':'__mod__',
    '__rmul__':'__mul__',
    '__ror__':'__or__',
    '__rpow__':'__pow__',
    '__rrshift__':'__rshift__',
    '__rsub__':'__sub__',
    '__rtruediv__':'__truediv__',
    '__rxor__':'__xor__',
    '__lt__':'__gt__',
    '__le__':'__ge__',
    '__eq__':'__eq__',
    '__ne__':'__ne__',
    '__gt__':'__lt__',
    '__ge__':'__le__',
}

_binary_magic_method_names = list(_binary_reflections.keys())

    
_other_magic_method_names = [
    '__abs__',
    '__neg__',
    '__pos__',
    #'__invert__',
    '__iadd__',
    '__iand__',
    '__idiv__',
    '__ifloordiv__',
    '__ilshift__',
    '__imod__',
    '__imul__',
    '__ior__',
    '__ipow__',
    '__irshift__',
    '__isub__',
    '__itruediv__',
    '__ixor__',
    '__complex__',
    '__contains__',
    '__delattr__',
    '__delete__',
    '__delitem__',
    '__delslice__',
    '__enter__',
    '__exit__',
    '__float__',
    '__get__',
    '__getitem__',
    '__getslice__',
    '__hash__',
    '__hex__',
    '__index__',
    '__int__',
    '__iter__',
    '__len__',
    '__long__',
    '__nonzero__',
    '__oct__',
    '__reversed__',
    '__set__',
    '__setitem__',
    '__setslice__',
    '__sub__',
    '__unicode__',
    '__str__',
]


def _set_magic_method(name):
    def magic_method(self, *args, **kwargs):
        return Intention(lambda x: getattr(self.function(x), name)(*_context_args(args)(x),
                                                                   **_context_kwargs(kwargs)(x)),
                         invert=self.inverted)

    return magic_method


for name in _other_magic_method_names:
    setattr(Intention, name, _set_magic_method(name))


def _set_reflected_magic_method(name):
    reflection = _binary_reflections[name]
    def magic_method(self, other):
        def context_wrapper(x):
            s = self.function(x) 
            o = other.function(x) if isinstance(other, Intention) else other
            if hasattr(s, name):
                return getattr(s, name)(o)
            elif hasattr(o, reflection):
                return getattr(o, reflection)(s)
            else:
                raise TypeError("unsupported operand type(s) for {0}: '{1}' and '{2}'".format(name, type(s), type(o)))
        return Intention(context_wrapper, invert=self.inverted)
    return magic_method

for name in _binary_magic_method_names:
    setattr(Intention, name, _set_reflected_magic_method(name))
    
# Initialize the global X symbol
X = Intention()