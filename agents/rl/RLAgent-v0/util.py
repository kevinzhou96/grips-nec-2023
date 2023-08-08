import os
import datetime

def format_time(seconds : float | int) -> str:
    periods = [
        ('d',   60*60*24),
        ('h',   60*60),
        ('m',   60),
        ('s',   1),
    ]
    strings = []
    for period, period_len in periods:
        if seconds >= period_len:
            period_value, seconds = divmod(seconds, period_len)
            strings.append(f"{int(period_value)}{period}")
    if seconds > 0:
        strings.append(f"{round(seconds*1000)} ms")
    return ' '.join(strings)
            
def get_dirname(path):
    return os.path.dirname(os.path.realpath(path))