from typing import List
from copy import deepcopy


class Solution:

    def __init__(self, assignments: dict[int, int]):
        self.assignments = assignments

    def __hash__(self):
        return hash(frozenset(self.assignments.items()))

    def is_partial(self) -> bool:
        return any(
            assigned_doctor is None
            for assigned_doctor
            in self.assignments.values()
        )

    def get_doctors_assignments(self, doctor_id: int) -> List[int]:
        return [
            shift_id
            for shift_id, assigned_doctor_id
            in self.assignments.items()
            if doctor_id == assigned_doctor_id
        ]

    def get_unassigned_shifts(self) -> List[int]:
        return [
            shift_id
            for shift_id, assigned_doctor_id
            in self.assignments.items()
            if assigned_doctor_id is None
        ]

    def assign(self, doctor: int, shift: int):
        """Generate a new partial solution by assigning a doctor to a shift"""
        new_assignments = deepcopy(self.assignments)
        new_assignments[shift] = doctor
        return Solution(new_assignments)
