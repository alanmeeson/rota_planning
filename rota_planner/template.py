from enum import Enum
from datetime import timedelta, date
from rota_planner.shift import Shift, ShiftType
from collections import defaultdict
from typing import List


class Weekday(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


class TemplateShift:
    def __init__(
            self,
            shift_type: ShiftType,
            start_time: timedelta,
            end_time: timedelta
    ):
        self.shift_type = shift_type
        self.start_time = start_time
        self.end_time = end_time

    def create_shift(self, shift_date: date) -> Shift:
        """Creates an actual shift on a specific date"""
        return Shift(
            self.shift_type,
            shift_date + self.start_time,
            shift_date + self.end_time
        )


class TemplateRota:

    def __init__(self):
        self._shifts = defaultdict(list)

    def add_shift(
            self,
            day: Weekday,
            shift_type: ShiftType,
            start_time: timedelta,
            end_time: timedelta,
            num_required: int
    ):

        for idx in range(num_required):
            shift = TemplateShift(shift_type, start_time, end_time)
            self._shifts[day].append(shift)

    def create_shifts(self, start_date: date, num_days: int) -> List[Shift]:
        """Create a list of actual shifts over a time range"""

        shifts = []

        for day in range(num_days):
            the_date = start_date + timedelta(days=day)
            the_day = the_date.isoweekday()
            the_days_shifts = self._shifts[the_day]

            for shift_template in the_days_shifts:
                shifts.append(shift_template.create_shift(the_date))

        return shifts
