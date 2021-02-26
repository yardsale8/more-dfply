from dfply import symbolic_evaluation
from more_itertools import flatten
from composable import pipeable

@symbolic_evaluation
def text_filter(col, pattern, *, case=False, regex=False, na=False):
    """ Create a text filter column a search term or regular expression.

    Use this function to replicate the functionality of a OpenRefine text filter.  
    The function is basically a wrapper around the `Series.str.contains` method, 
    but returns a boolean dtype Series by default; where `np.nan` are evaluated 
    as `False` by default.

    Note that method is equivalent to `str.contains` or `re.find` and will return 
    true if `pattern` is found anywhere in the string.

    Args:
        col (pandas.Series or dfply.Intention): Filter based on this column.
        pattern (str): Search for this `pattern`.
        case (bool): Toggle case-sensitivity. Default = False
        regex (bool): If the pattern should be considered a regular expression. Default = False
        na (bool): Value used to replace `np.nan`.  Default = False to guarantee a bool dtype

    Returns:
        Either a pd.Series [bool dtype by default] 
        or a dfply.Intention which will be evaluated by the host function.

    """
    return col.str.contains(pattern, case=case, regex=regex, na=na)

def get_text_facets(col):
    cnts = col.value_counts()
    return [(l, v) for l, v in zip(cnts.index, cnts)]


@symbolic_evaluation
def text_facet(col, *args):
    labels = [s for s in args if isinstance(s, str)] + [v for l in args if not isinstance(l, str) for v in l if isinstance(v, str)]
    return col.isin(labels) 


@pipeable
def _between(from_, to, val):
    return (from_ == 'min' or val >= from_) and (to == 'max' or val <= to)


def _facet_by_label_count(col, from_ = 'min', to = 'max'):
    cnts = get_text_facets(col)
    lbls_to_keep = [l for l, c in cnts if between(from_, to, c)]
    return col.isin(lbls_to_keep)


def facet_by_label_count(col, from_ = 'min', to = 'max'):
    if isinstance(col, Intention):
        return Intention(lambda df: _facet_by_label_count(col.evaluate(df), from_, to))
    else: 
        return _facet_by_label_count(col, from_, to)