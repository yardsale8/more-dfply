from collections import defaultdict
from dfply import make_symbolic, pipe, symbolic_evaluation, Intention, dfpipe, rename, flatten, X
from dfply.base import Intention
import pandas as pd
import numpy as np
import re
from string import punctuation, whitespace
from composable import pipeable
from functools import reduce
from toolz import identity, last

__all__ = ("fix_names",
           "extract",
           "to_datetime",
           "recode",
           "set_index",
           "row_index_slice",
           "maybe_tile",
           "maybe_eval",
           "any_intention",
           "tiled_where",
           "cond_eval",
           "ifelse",
           "maybe_combine",
           "coalesce",
           "case_when",
           "col_zip",
           "union_all",

          )


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


@dfpipe
def fix_names(df, make_lower=False):
    """ Creates a dict of new_name:old_name pairs.
    
    Any name that is not an identifier (using the new .isidentifier predicate) 
    The new names have all punctuation removed, outer whitespace removed,
    and whitespace replaced with _.
    """
    return df >> rename(**{fix_name(col, make_lower=make_lower):col for col in df.columns})


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
@symbolic_evaluation(eval_as_selector=True)
def set_index(df, col, drop = True):
    return df.set_index(col, drop = drop)


@make_symbolic
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

    

@pipeable
def maybe_eval(df, col): 
    """ Evaluate col with df context whenever col is an Intention"""
    return col.evaluate(df) if hasattr(col, 'evaluate') else col


def any_intention(*args): 
    """Flattens args and checks for Intentions"""
    return any(hasattr(o, 'evaluate') for o in flatten(args))


def tiled_where(cond, then, else_):
    """Make then and else_ the same length as cond then perform np.where"""
    n = len(cond)
    return np.where(cond, maybe_tile(n, then), maybe_tile(n, else_))


def cond_eval(cond, expr, df):
    """ Only evaluate expr, which maybe an Intention, when cond == True"""
    return maybe_eval(df, expr) if cond else None


def symbolic_ifelse(cond, then, else_):
    """ Returns a Series that is the same length as cond, picking elements from then and else_ 
    based on the truth of cond.
    
    If then or else_ are instances of Intention, then they will only be evaluated when needed.
    """
    def outfunc(df): 
        cond_out = maybe_eval(df, cond)
        n = len(cond_out)
        if cond_out.all():
            return maybe_eval(df, then) >> maybe_tile(n)
        elif not cond_out.any():
            return maybe_eval(df, else_) >> maybe_tile(n)
        else:
            t = maybe_eval(df, then)
            e = maybe_eval(df, else_)
            return pd.Series(tiled_where(cond_out, t, e))
    return Intention(outfunc)


def ifelse(cond, then, else_):
    """ Returns a Series that is the same length as cond, picking elements from then and else_ 
        based on the truth of cond.
        
        If then or else_ are instances of Intention, then they will only be evaluated when needed."""
    if any_intention([cond, then, else_]):
        return symbolic_ifelse(cond, then, else_)
    else:
        return pd.Series(tiled_where(cond, then, else_))


def maybe_combine(acc, col):
    return acc.combine_first(col) if acc.isna().any() else acc


@make_symbolic
def coalesce(*args):
    return reduce(maybe_combine, args)
 
    

# def ifelse_and_coalesce(args, apply_first=identity, default=np.nan):
#     return coalesce(*[apply_first(ifelse(c, t, default)) for c, t in args])

def get_length(col):
    if (isinstance(col, str) # Treat strings as singletons
        or not hasattr(col, '__len__') # Other singletons
       ):
        return 1
    else:
        return len(col)



def get_RHS(case, then, n):
    if isinstance(case, bool) and case:
        return maybe_tile(n, then)
    elif isinstance(case, bool):
        return maybe_tile(n, np.nan)
    else:
        return ifelse(case, then, np.nan)


@make_symbolic
def case_when(*args, default=np.nan):
    lengths = ([get_length(t) for c, t in args] 
              + [get_length(c) for c, t in args])
    n = max(lengths)
    assert all(l == 1 or l == n for l in lengths), "All LHS and RHS need to be singletons or the same length."
    rhs = [get_RHS(c, t, n) for c, t in args]
    return coalesce(*rhs)


@make_symbolic
def col_zip(col1, col2, *cols):
    """ zip two or more columns into a column of tuples
    
    This function is useful when mapping two values to 
    another using a dictionary, 
    i.e. (lat, long) -> address
    """
    return pd.Series(zip(col1, col2, *cols))


@dfpipe
def union_all(left_df, right_df, ignore_index=True):
    """Union two data frames, keeping all duplicate rows.
    
       Note that columns are matched by potition
    """
    return pd.concat([left_df, right_df], ignore_index=ignore_index)

