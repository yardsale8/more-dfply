from collections import defaultdict
from dfply import make_symbolic, pipe, symbolic_evaluation, Intention, dfpipe, rename, flatten, X
import pandas as pd
import numpy as np
import re
from string import punctuation, whitespace
from functoolz import pipeable
from functools import reduce
from toolz import identity, last
from unpythonic import unfold

STARTS_WITH_DIGITS_REGEX = re.compile(r'\d+')
PUNC_REGEX = re.compile('[{0}]'.format(re.escape(punctuation.replace('_', ''))))
WS_REGEX = re.compile('[{0}]'.format(re.escape(whitespace)))

def fix_name(name:str, make_lower:bool=False) -> str:
    """ Makes a column name a proper identifier.
    
    This function will
    
    1. Strip leading/trailing whitspace
    2. Remove all puncuation except _
    3. Add a _ to the start of any string that start with a digit
    """
    strip = pipeable(lambda s: s.strip())
    remove_punc = pipeable(lambda s: PUNC_REGEX.sub('',s))
    fix_starting_digit = pipeable(lambda s: '_' + s if STARTS_WITH_DIGITS_REGEX.match(s) else s)
    replace_whitespace = pipeable(lambda s: WS_REGEX.sub('_', s))
    if name.isidentifier():
        return name
    else:
        new_name = name >> strip >> remove_punc >> fix_starting_digit >> replace_whitespace
        return new_name.lower() if make_lower else new_name

def test_fix_name() -> None:
    good_name = 'good_name'
    bad_name = " 1bad_name !1!\n"
    assert fix_name(good_name).isidentifier()
    assert fix_name(bad_name).isidentifier()
test_fix_name()
    

@dfpipe
def fix_names(df, make_lower=False):
    """ Creates a dict of new_name:old_name pairs.
    
    Any name that is not an identifier (using the new .isidentifier predicate) 
    The new names have all punctuation removed, outer whitespace removed,
    and whitespace replaced with _.
    """
    return df >> rename(**{fix_name(col, make_lower=make_lower):col for col in df.columns})


def test_fix_names():
    good_name = 'good_name'
    bad_name = " 1bad_name!1!\n"
    df = pd.DataFrame({good_name: [1,2,3],
                       bad_name:[1,2,3]})
    assert all(k.isidentifier() for k in df >> fix_names)
test_fix_names()


@make_symbolic
def extract(col, pattern):
    if re.compile(pattern).groups != 1:
        raise ValueError('extract requires a pattern with exactly 1 group')
    return col.str.extract(pattern, expand=False)


@make_symbolic
def to_datetime(series, infer_datetime_format=True):
    return pd.to_datetime(series, infer_datetime_format=infer_datetime_format)


@make_symbolic
def recode(col, d, default=None):
    if default is not None:
        new_d = defaultdict(lambda: default)
        new_d.update(d)
        d = new_d
    return col.map(d)


@pipe
@symbolic_evaluation(eval_as_selector=[1])
def set_index(df, col, drop = True):
    return df.set_index(col, drop = drop)


@pipe
@symbolic_evaluation
def row_index_slice(df, *args):
    assert len(args) in (1,2), "loc requires 1-2 arguments"
    if len(args) == 1:
        return df.loc[args[0]]
    else:
        return df.loc[args[0]:args[1]]
    

@pipeable
def maybe_tile(n, col):
    if (isinstance(col, str) # Treat strings as singletons
        or not hasattr(col, '__len__') # Other singletons
        or len(col) < n # Too short
       ):
        return pd.Series(np.tile(col, n)[:n])
    else:
        return pd.Series(col[:n])

    
def test_maybe_tile():
    # Test singletons
    for val in (5, 'singleton', 2.3):
        assert len(maybe_tile(3, val)) == 3
        assert (maybe_tile(3, val) == val).all()
    # Test piping
    assert (5 >> maybe_tile(2) == pd.Series([5,5])).all()
    # Test tiling for too short
    assert (maybe_tile(5, [1,2]) == pd.Series([1,2,1,2,1])).all()
    # Test truncating for too long
    assert (maybe_tile(3, range(10)) == pd.Series(range(3))).all()
test_maybe_tile()


@pipeable
def maybe_eval(df, col): 
    """ Evaluate col with df context whenever col is an Intention"""
    return col.evaluate(df) if isinstance(col, Intention) else col


def test_maybe_eval():
    assert maybe_eval(None, 5) == 5
    assert maybe_eval(5, X) == 5
    assert maybe_eval(5.0, X.is_integer())
    f = make_symbolic(lambda x: x + 1)
    assert maybe_eval([1,2,3], f(X[0])) == 2
test_maybe_eval()


def any_intention(*args): 
    """Flattens args and checks for Intentions"""
    return any(isinstance(o, Intention) for o in flatten(args))


def test_any_intention():
    assert any_intention(X)
    assert any_intention([X])
    assert any_intention([[X]])
    assert not any_intention(1)
    assert not any_intention(1, 'a', 5.5)
    assert not any_intention(1, 'a', [5.5])
test_any_intention()   


def tiled_where(cond, then, else_):
    """Make then and else_ the same length as cond then perform np.where"""
    n = len(cond)
    return np.where(cond, maybe_tile(n, then), maybe_tile(n, else_))


def test_tiled_where():
    assert (tiled_where([True, True], 'Yes', 'No') == 'Yes').all()
    assert (tiled_where([False, False], 'Yes', 'No') == 'No').all()
    assert (tiled_where([True, False], 'Yes', 'No') == pd.Series(['Yes', 'No'])).all()
    assert (tiled_where([True, True], [1,2], [3,4]) == pd.Series([1,2])).all()
    assert (tiled_where([False, False], [1,2], [3,4]) == pd.Series([3,4])).all()
    assert (tiled_where([True, False], [1,2], [3,4]) == pd.Series([1,4])).all()
    assert (tiled_where([False, True], [1,2], [3,4]) == pd.Series([3,2])).all()
    assert (tiled_where([True, True], [1,2,3], [4, 5, 6]) == pd.Series([1,2])).all()
    assert (tiled_where([False, False], [1,2,3], [4, 5, 6]) == pd.Series([4,5])).all()
test_tiled_where()


def cond_eval(cond, expr, df):
    """ Only evaluate expr, which maybe an Intention, when cond == True"""
    return maybe_eval(df, expr) if cond else None


def test_cond_eval():
    assert cond_eval(True, X, 5) == 5
    assert cond_eval(True, X.is_integer(), 5.0)
    f = make_symbolic(lambda x: x + 1)
    assert cond_eval(True, f(X), 5) == 6
    assert cond_eval(False, X/0, 5) is None
test_cond_eval()


def ifelse(cond, then, else_):
    """ Returns a Series that is the same length as cond, picking elements from then and else_ 
        based on the truth of cond.
        
        If then or else_ are instances of Intention, then they will only be evaluated when needed."""
    def outfunc(df): 
        cond_out = maybe_eval(df, cond)
        then_out = cond_eval(cond_out.any(), then, df)
        else_out = cond_eval(not cond_out.all(), else_, df)
        return pd.Series(tiled_where(cond_out, then_out, else_out))
    return Intention(outfunc) if any_intention(cond, then, else_) else pd.Series(tiled_where(cond, then, else_))

def test_ifelse():
    all_true = np.repeat(True, 3)
    all_false = np.repeat(False, 3)
    assert (ifelse(all_true, 'Yes', 'No') == 'Yes').all()
    assert (ifelse(all_false, 'Yes', 'No') == 'No').all()
    s1 = pd.Series(np.arange(1, 4, 1))
    s2 = pd.Series(np.arange(11, 14, 1))
    assert (ifelse(all_true, s1, s2) == s1).all()
    assert (ifelse(all_false, s1, s2) == s2).all()
    short = np.arange(1,3,1)
    long = np.arange(1,5,1)
    assert (ifelse(all_true, short, long) == pd.Series([1,2,1])).all()
    assert (ifelse(all_false, short, long) == pd.Series([1,2,3])).all()
    df = pd.DataFrame({'cat':['a', 'a', 'b', 'b', 'b'],
                   'v1':[1,2,1,2,3],
                   'v2':[3,4,3,4,5] })
    assert (ifelse(df.cat == 'a', df.v1, df.v2) == pd.Series([1, 2, 3, 4, 5])).all()
    e1 = ifelse(df.cat == 'a', X.v1, df.v2)
    e2 = ifelse(df.cat == 'a', df.v1, X.v2)
    e3 = ifelse(df.cat == 'a', X.v1, X.v2)
    e4 = ifelse(X.cat == 'a', X.v1, X.v2)
    assert (e1.evaluate(df) == pd.Series([1, 2, 3, 4, 5])).all()
    assert (e2.evaluate(df) == pd.Series([1, 2, 3, 4, 5])).all()
    assert (e3.evaluate(df) == pd.Series([1, 2, 3, 4, 5])).all()
    assert (e4.evaluate(df) == pd.Series([1, 2, 3, 4, 5])).all()
test_ifelse()
                     

@pipeable
def maybe_combine(acc, col, apply_first=identity):
    acc = apply_first(acc)
    return acc.combine_first(apply_first(col)) if acc.isna().any() else acc

def test_maybe_combine():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    s1 = pd.Series([1,      2,      np.nan, np.nan])
    s2 = pd.Series([np.nan, 1,      2,      np.nan])
    s3 = pd.Series([np.nan, np.nan, 1,      2])
    o1 = pd.Series([1, 2, 2, np.nan])
    o2 = pd.Series([1, 2, 1, 2])
    assert equal_or_na(maybe_combine(s1, s2), o1)
    assert equal_or_na(maybe_combine(s1, s3), o2)
    df = pd.DataFrame({'c1':[1,      2,      np.nan, np.nan],
                       'c2':[np.nan, 1,      2,      np.nan],
                       'c3':[np.nan, np.nan, 1,      2]})
    assert equal_or_na(maybe_combine(X.c1, 
                                     X.c2, 
                                     maybe_eval(df)),
                       o1)
    assert equal_or_na(maybe_combine(X.c1, 
                                     X.c3, 
                                     maybe_eval(df)),
                       o2)
test_maybe_combine()


def combine_all(args, apply_first=identity):
    return reduce(maybe_combine(apply_first=apply_first), args)


def test_combine_all():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    s1 = pd.Series([1,      2,      np.nan, np.nan])
    s2 = pd.Series([np.nan, 1,      2,      np.nan])
    s3 = pd.Series([np.nan, np.nan, 1,      2])
    o1 = pd.Series([1, 2, 2, np.nan])
    o2 = pd.Series([1, 2, 1, 2])
    o3 = pd.Series([1, 2, 2, 2])
    assert equal_or_na(combine_all([s1, s2]), o1)
    assert equal_or_na(combine_all([s1, s3]), o2)
    assert equal_or_na(combine_all([s1, s2, s3]), o3)
    df = pd.DataFrame({'c1':[1,      2,      np.nan, np.nan],
                       'c2':[np.nan, 1,      2,      np.nan],
                       'c3':[np.nan, np.nan, 1,      2]})
    assert equal_or_na(combine_all([X.c1, 
                                    X.c2], 
                                    maybe_eval(df)),
                       o1)
    assert equal_or_na(combine_all([X.c1, 
                                    X.c3], 
                                    maybe_eval(df)),
                       o2)
    assert equal_or_na(combine_all([X.c1, 
                                    X.c2,
                                    X.c3], 
                                    maybe_eval(df)),
                       o3)
test_combine_all()


@pipeable
def arg_eval_and_combine(args, df):
    return combine_all(args, apply_first=maybe_eval(df))

def test_arg_eval_and_combine():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    o1 = pd.Series([1, 2, 2, np.nan])
    o2 = pd.Series([1, 2, 1, 2])
    o3 = pd.Series([1, 2, 2, 2])
    df = pd.DataFrame({'c1':[1,      2,      np.nan, np.nan],
                       'c2':[np.nan, 1,      2,      np.nan],
                       'c3':[np.nan, np.nan, 1,      2]})
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c2], 
                                             df),
                       o1)
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c3], 
                                             df),
                       o2)
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c2,
                                             X.c3], 
                                             df),
                       o3)
    # Test currying 
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c2])(df),
                       o1)
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c3])(df),
                       o2)
    assert equal_or_na(arg_eval_and_combine([X.c1, 
                                             X.c2,
                                             X.c3])(df),
                       o3)
    # Test piping
    assert equal_or_na(df >> arg_eval_and_combine([X.c1, X.c2]),
                       o1)
    assert equal_or_na(df >> arg_eval_and_combine([X.c1, X.c3]),
                       o2)
    assert equal_or_na(df >> arg_eval_and_combine([X.c1, X.c2, X.c3]),
                       o3)
test_arg_eval_and_combine()


def coalesce(*args):
    return Intention(arg_eval_and_combine(args)) if any_intention(args) else combine_all(args)
 
    
def test_arg_eval_and_combine():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    o1 = pd.Series([1, 2, 2, np.nan])
    o2 = pd.Series([1, 2, 1, 2])
    o3 = pd.Series([1, 2, 2, 2])
    df = pd.DataFrame({'c1':[1,      2,      np.nan, np.nan],
                       'c2':[np.nan, 1,      2,      np.nan],
                       'c3':[np.nan, np.nan, 1,      2]})
    assert equal_or_na(coalesce(df.c1, df.c2), o1)
    assert equal_or_na(coalesce(df.c1, df.c3), o2)
    assert equal_or_na(coalesce(df.c1, df.c2, df.c3), o3)
    assert equal_or_na(coalesce(X.c1, X.c2).evaluate(df), o1)
    assert equal_or_na(coalesce(X.c1, X.c3).evaluate(df), o2)
    assert equal_or_na(coalesce(X.c1, X.c2, X.c3).evaluate(df), o3)
test_arg_eval_and_combine()


def ifelse_and_coalesce(args, apply_first=identity):
    return coalesce(*[apply_first(ifelse(c, t, np.nan)) for c, t in args])


@pipeable
def eval_and_case_when(args, df):
    return ifelse_and_coalesce(args, apply_first=maybe_eval(df))


def case_when(*args):
    return Intention(eval_and_case_when(args)) if any_intention(args) else ifelse_and_coalesce(args)


def test_case_when():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    df = pd.DataFrame({'cat':['a','a','b','b','b','c','c', 'd','d'],
                       'val':[ 1, 2, 1, 2, 3, 1, 2, 1, 2]})
    o1 = pd.Series([2, 3] + 7*[np.nan])
    o2 = pd.Series([2, 3, 3, 4, 5] + 4*[np.nan])
    o3 = pd.Series([2, 3, 3, 4, 5, 4, 5] + 2*[np.nan]) 
    assert equal_or_na(case_when((df.cat == 'a', df.val + 1)), 
                       o1)
    assert equal_or_na(case_when((df.cat == 'a',  df.val + 1), 
                                 (df.cat == 'b', df.val + 2)), 
                       o2)
    assert equal_or_na(case_when((df.cat == 'a', df.val + 1), 
                                 (df.cat == 'b', df.val + 2), 
                                 (df.cat == 'c', df.val + 3)), 
                       o3) 
    assert equal_or_na(case_when((X.cat == 'a', 
                                  X.val + 1)).evaluate(df), 
                       o1)
    assert equal_or_na(case_when((X.cat == 'a', X.val + 1), 
                                 (X.cat == 'b', X.val + 2)).evaluate(df) , 
                       o2)
    assert equal_or_na(case_when((X.cat == 'a', X.val + 1), 
                                 (X.cat == 'b', X.val + 2), 
                                 (X.cat == 'c', X.val + 3)).evaluate(df), 
                       o3)
test_case_when()


@make_symbolic
def col_zip(col1, col2, *cols):
    """ zip two or more columns into a column of tuples
    
    This function is useful when mapping two values to 
    another using a dictionary, 
    i.e. (lat, long) -> address
    """
    return pd.Series(zip(col1, col2, *cols))


def test_col_zip():
    G1 = 3*['a'] + 3*['b']
    G2 = 2*['u'] + 2*['v'] + 2*['w']
    V = list(range(6))
    df = pd.DataFrame({'G1':G1,
                       'G2':G2,
                       'V':V})
    assert (col_zip(df.G1, df.G2) == pd.Series(zip(df.G1, G2))).all()
    assert (col_zip(df.G1, df.G2, df.V) == pd.Series(zip(df.G1, G2, V))).all()
    e1 = col_zip(X.G1, X.G2)
    e2 = col_zip(X.G1, X.G2, X.V)
    assert (e1.evaluate(df) == pd.Series(zip(df.G1, G2))).all()
    assert (e2.evaluate(df) == pd.Series(zip(df.G1, G2, V))).all()
test_col_zip()


@dfpipe
def union_all(left_df, right_df, ignore_index=True):
    """Union two data frames, keeping all duplicate rows.
    
       Note that columns are matched by potition
    """
    return pd.concat([left_df, right_df], ignore_index=ignore_index)


def test_union_all():
    df1 = pd.DataFrame({'a': [1,2,3],
                        'b': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'a': [1,2,3],
                        'b': ['a', 'b', 'c']})
    df_out = df1 >> union_all(df2)
    a_out = pd.Series([1, 2, 3, 1, 2, 3])
    b_out = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'])
    assert (df_out.a == a_out).all()
    assert (df_out.b == b_out).all()
test_union_all()