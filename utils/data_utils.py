def normalize_str_series(series):
    """Normalize a pandas Series: convert to string, lowercase, and strip whitespace."""
    return series.astype(str).str.lower().str.strip() 