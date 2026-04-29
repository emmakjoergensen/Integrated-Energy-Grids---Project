def make_profile(series, target_cf):
    series = series.fillna(0)
    profile = series / series.max()
    profile = profile / profile.mean() * target_cf
    return profile.clip(upper=1.0)