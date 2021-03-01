from more_dfply import __version__
from more_dfply.more_dfply import fix_name, fix_names, maybe_tile, maybe_eval, any_intention, tiled_where, cond_eval, ifelse, maybe_combine, coalesce, case_when, col_zip, union_all
import pandas as pd
import numpy as np
from dfply import make_symbolic, pipe, symbolic_evaluation, Intention, dfpipe, rename, flatten, X


def test_version():
    assert __version__ == '0.1.0'


def test_fix_name() -> None:
    good_name = 'good_name'
    bad_name = " 1bad_name !1!\n"
    assert fix_name(good_name).isidentifier()
    assert fix_name(bad_name).isidentifier()


def test_fix_names():
    good_name = 'good_name'
    bad_name = " 1bad_name!1!\n"
    df = pd.DataFrame({good_name: [1,2,3],
                       bad_name:[1,2,3]})
    assert all(k.isidentifier() for k in df >> fix_names)


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


def test_maybe_eval():
    assert maybe_eval(None, 5) == 5
    assert maybe_eval(5, X) == 5
    assert maybe_eval(5.0, X.is_integer())
    f = make_symbolic(lambda x: x + 1)
    assert maybe_eval([1,2,3], f(X[0])) == 2


def test_any_intention():
    assert any_intention(X)
    assert any_intention([X])
    assert any_intention([[X]])
    assert not any_intention(1)
    assert not any_intention(1, 'a', 5.5)
    assert not any_intention(1, 'a', [5.5])


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


def test_cond_eval():
    assert cond_eval(True, X, 5) == 5
    assert cond_eval(True, X.is_integer(), 5.0)
    f = make_symbolic(lambda x: x + 1)
    assert cond_eval(True, f(X), 5) == 6
    assert cond_eval(False, X/0, 5) is None


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
                     


def test_maybe_combine():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    s1 = pd.Series([1,      2,      np.nan, np.nan])
    s2 = pd.Series([np.nan, 1,      2,      np.nan])
    s3 = pd.Series([np.nan, np.nan, 1,      2])
    o1 = pd.Series([1, 2, 2, np.nan])
    o2 = pd.Series([1, 2, 1, 2])
    assert equal_or_na(maybe_combine(s1, s2), o1)
    assert equal_or_na(maybe_combine(s1, s3), o2)


def test_coalesce():
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


def test_case_when():
    equal_or_na = lambda s1, s2: ((s1 == s2) | pd.isna(s1)).all()
    df = pd.DataFrame({'cat':['a','a','b','b','b','c','c', 'd','d'],
                       'val':[ 1, 2, 1, 2, 3, 1, 2, 1, 2]})
    sing1 = pd.Series(2*[1] + 7*[np.nan])
    sing2 = pd.Series(2*[1] + 3*[2] + 4*[np.nan])
    sing3 = pd.Series([1, 1, 3, 4, 5] + 4*[np.nan])
    sing4 = pd.Series(2*[1] + 3*[2] + 4*[3])
    o1 = pd.Series([2, 3] + 7*[np.nan])
    o2 = pd.Series([2, 3, 3, 4, 5] + 4*[np.nan])
    o3 = pd.Series([2, 3, 3, 4, 5, 4, 5] + 2*[np.nan]) 
    assert equal_or_na(case_when((df.cat == 'a', df.val + 1)), 
                       o1)
    assert equal_or_na(case_when((df.cat == 'a', 1)), 
                       sing1)
    assert equal_or_na(case_when((df.cat == 'a',  df.val + 1), 
                                 (df.cat == 'b', df.val + 2)), 
                       o2)
    assert equal_or_na(case_when((df.cat == 'a', 1), 
                                 (df.cat == 'b', 2)), 
                       sing2)
    assert equal_or_na(case_when((df.cat == 'a', 1), 
                                 (df.cat == 'b', df.val + 2)), 
                       sing3)
    assert equal_or_na(case_when((df.cat == 'a', df.val + 1), 
                                 (df.cat == 'b', df.val + 2), 
                                 (df.cat == 'c', df.val + 3)), 
                       o3) 
    assert equal_or_na(case_when((df.cat == 'a', 1), 
                                 (df.cat == 'b', 2), 
                                 (True, 3)), 
                       sing4)
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