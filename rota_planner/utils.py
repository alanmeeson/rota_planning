from datetime import datetime


def calc_overlap(
        s1: datetime,
        s2: datetime,
        e1: datetime,
        e2: datetime
) -> float:
    """Calculates the overlap in hours between two time ranges

    Args:
        s1, e1: Start/End times of range 1
        s2, e2: Start/End times of range 2
    Return:
        overlap as a number of hours
    """
    latest_start = max(s1, s2)
    earliest_end = min(e1, e2)
    delta = (earliest_end - latest_start)

    return max(0, delta.total_seconds() // 3600)
