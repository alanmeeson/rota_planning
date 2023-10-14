import numpy as np
from enum import Enum
from datetime import datetime, timedelta
from typing import List
from itertools import product

from rota_planner.utils import calc_overlap


class ShiftType(Enum):
    """The types of shift we care about"""
    ZERO = 0  # Zero Day/EWTD/Off
    STANDARD = 1  # Standard day shift
    ONCALL = 2  # Oncall or long shift
    NIGHT = 3  # Night shift


class Shift:
    """A shift to fill"""

    def __init__(self, shift_type: ShiftType, start_time: datetime, end_time: datetime):
        self.type = shift_type
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"{self.type} Shift: {self.start_time} to {self.end_time}"

    def time_between(self, shift) -> float:
        """Calculates time between two shifts in hours"""

        if self.start_time < shift.start_time:
            return (shift.start_time - self.end_time).total_seconds() // 3600
        else:
            return (self.start_time - shift.end_time).total_seconds() // 3600

    def is_night_shift(self) -> bool:
        # Yes: if more than 3 hours between 23:00 - 06:00
        return self.type == ShiftType.NIGHT

    def is_weekend_shift(self) -> bool:
        #yes if any work between 00:01 sat and 23:59 sun

        # As a hack, just check if the start or end is on a weekend day.
        # TODO: consider edge cases
        return (self.start_time.isoweekday() in {6, 7}) or (self.end_time.isoweekday() in {6, 7})

    def can_take_leave(self) -> bool:
        return self.type in {ShiftType.ZERO, ShiftType.STANDARD}

    def duration(self) -> float:
        """The duration of the shift in hours"""
        return (self.end_time - self.start_time).total_seconds() // 3600


def min_time_between_shifts(shifts: List[Shift]) -> float:
    """Calculates the minimum gap in hours between a set of shifts"""

    return min(shifts[a].time_between(shifts[b]) for a, b in product(range(len(shifts)), repeat=2) if a != b)


def may_follow_night_shift(night_shift, shift):
    if shift.type == ShiftType.NIGHT:
        # TODO: Want to check for max 4
        return True
    else:
        # Require at least 46 hours rest after night shifts.
        return night_shift.time_between(shift) > 46


def may_follow_oncall_shift(oncall_shift, shift):
    if (shift.type == ShiftType.ONCALL) and not (
            oncall_shift.is_weekend_shift() and
            shift.is_weekend_shift()
    ):
        # May not have two consecutive on call shifts
        # except on the weekend

        return False
    elif (shift.type != ShiftType.ONCALL) and shift.duration() > 10:
        # Day after oncall shift must not exceed 10 hours
        return False
    else:
        return True


def max_hours_in_period(shifts: List[Shift], period: int):
    hours = 0

    for shift in shifts:
        start_time = shift.start_time
        end_time = shift.start_time + timedelta(hours=period)

        overlapping_shifts_duration = sum([
            shift.duration() for shift in shifts
            if calc_overlap(start_time, shift.start_time, end_time, shift.end_time) > 0
        ])

        hours = max(hours, overlapping_shifts_duration)

    return hours


def min_hours_in_period(shifts: List[Shift], period: int):
    hours = np.inf

    for shift in shifts:
        start_time = shift.start_time
        end_time = shift.start_time + timedelta(hours=period)

        overlapping_shifts_duration = sum([
            shift.duration() for shift in shifts
            if calc_overlap(start_time, shift.start_time, end_time, shift.end_time) > 0
        ])

        hours = min(hours, overlapping_shifts_duration)

    return hours
