from typing import List
from datetime import date, datetime
from collections import Counter
from rota_planner.utils import calc_overlap
from rota_planner.shift import Shift, min_time_between_shifts, may_follow_night_shift, max_hours_in_period


class Preference:
    """A Day the Doctor wants to be not working on."""
    def __init__(self, day: date):
        """Create a preference from a day."""
        self.date = day
        self.start_time = datetime.combine(day, datetime.min.time())
        self.end_time = datetime.combine(day, datetime.max.time())

    def __str__(self):
        return f"Requested Off: {self.start_time.date()}"

    def is_clash(self, shift) -> bool:
        """Return True if the proposed shift clashes with this preference

        Args:
            shift: a Shift object to check
        """
        clashes = calc_overlap(
            self.start_time, shift.start_time,
            self.end_time, shift.end_time
        ) > 0

        return clashes and not shift.can_take_leave()


class Doctor:
    def __init__(self, name: str, preferences: List[Preference] = None):
        self.name = name

        if not preferences:
            preferences = list()

        self.preferences = preferences

    def add_preference(self, day: date):
        self.preferences.append(Preference(day))

    def is_clash(self, shift: Shift) -> bool:
        """Check if the shift clashes with any of the Doctor's Preferences"""
        return any(
            preference.is_clash(shift)
            for preference in self.preferences
        )

    def is_valid(self, shifts: List[Shift]) -> bool:
        """Check if the proposed rota for this doctor is valid.
        Args:
            shifts: the list of shifts the doctor has been assigned
        Returns: true if it doesn't break any rules.
        """
        # TODO: add the other constraints
        valid = True

        # Minimum of 11 hours between shifts
        if len(shifts) > 1:
            valid = valid and (min_time_between_shifts(shifts) >= 11)

        # Maximum of 72 hours in any 168 hour period (ie: 1 week)
        valid = valid and (max_hours_in_period(shifts, 7*24) <= 72)

        # Maximum of 48 hours per week on average
        valid = valid and ((max_hours_in_period(shifts, 8*7*24) / 8) <= 48)

        return valid

    def preference_score(self, shifts: List[Shift]) -> float:
        """Calculates how badly this assignment violates the doctors preferences.

        Args:
            shifts: The doctor's shifts
        Returns: float indicating what proportion of their preferences are violated.
        """

        clashes = [self.is_clash(shift) for shift in shifts]
        num_clashes = len(clashes)

        return num_clashes / len(self.preferences)



