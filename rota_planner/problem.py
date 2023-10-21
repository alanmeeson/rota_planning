import numpy as np
from typing import List
from rota_planner.shift import Shift
from rota_planner.doctor import Doctor
from rota_planner.rota import Solution
from tqdm import tqdm
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    inv_depth: float=field(compare=False)
    depth: int=field(compare=False)
    item: Any=field(compare=False)


class Problem:

    def __init__(self, shifts: List[Shift], doctors: List[Doctor]):
        self.shifts = shifts
        self.doctors = doctors
        self._num_weekend_shifts = len([shift for shift in self.shifts if shift.is_weekend_shift()])
        self._num_hours = sum(shift.duration() for shift in self.shifts)
        self._num_night_shifts = len([shift for shift in self.shifts if shift.is_night_shift()])
        self._min_dissapointment = self.calc_minimum_disapointment()
        self.current_llb_solution = None
        self.current_best = None

    def create_empty_solution(self):
        return Solution({idx: None for idx, shift in enumerate(self.shifts)})

    def get_doctors_rota(self, rota: Solution, doctor_id: int) -> List[Shift]:

        return [
            self.shifts[shift_id]
            for shift_id
            in rota.get_doctors_assignments(doctor_id)
        ]

    def score(self, rota: Solution) -> float:
        """Returns the score for a rota based on violated preferences"""

        score = 0

        # Score preference violations
        preference_score = 0
        max_num_weekends = 0
        max_hours = 0
        max_num_nights = 0

        for doctor_id, doctor in enumerate(self.doctors):
            doctor_shifts = self.get_doctors_rota(rota, doctor_id)
            doctor_score = doctor.preference_score(doctor_shifts)
            preference_score += doctor_score

            num_weekend_shifts = len([shift for shift in doctor_shifts if shift.is_weekend_shift()])
            max_num_weekends = max(max_num_weekends, num_weekend_shifts)

            num_hours = sum(shift.duration() for shift in doctor_shifts)
            max_hours = max(max_hours, num_hours)

            num_night_shifts = len([shift for shift in doctor_shifts if shift.is_night_shift()])
            max_num_nights = max(max_num_nights, num_night_shifts)

        # Normalise to roughly 0-1
        preference_score = preference_score / len(self.doctors)
        max_num_weekends = max_num_weekends / self._num_weekend_shifts
        max_hours = max_hours / (self._num_hours / len(self.doctors))
        max_num_nights = max_num_nights / self._num_night_shifts

        score = preference_score + max_num_weekends + max_hours + max_num_nights
        return score

    def lower_bound(self, rota: Solution) -> float:
        """Calculate the best possible score for the partial solution
        """

        # Score preference violations
        preference_score = 0
        max_num_weekends = 0
        max_hours = 0
        max_num_nights = 0
        for doctor_id, doctor in enumerate(self.doctors):
            # We assume that other than existing violations, doctors preferences won't be violated.
            doctor_shifts = self.get_doctors_rota(rota, doctor_id)
            doctor_score = doctor.preference_score(doctor_shifts)
            preference_score += doctor_score

            # We assume that other than already assigned weekends,  all other weekends will be distributed evenly
            num_weekend_shifts = len([shift for shift in doctor_shifts if shift.is_weekend_shift()])
            max_num_weekends = max(max_num_weekends, num_weekend_shifts)

            # We assume that other than already assigned hours, all other hours will be distributed evenly
            num_hours = sum(shift.duration() for shift in doctor_shifts)
            max_hours = max(max_hours, num_hours)

            # We assume that other than already assigned nights,  all other nights will be distributed evenly
            num_night_shifts = len([shift for shift in doctor_shifts if shift.is_night_shift()])
            max_num_nights = max(max_num_nights, num_night_shifts)

        # Distribute remaining weekends evenly
        # TODO: is this right?
        #unassigned_shifts = rota.get_unassigned_shifts()
        #unassigned_weekends = len([shift_idx for shift_idx in unassigned_shifts if self.shifts[shift_idx].is_weekend_shift()])
        #max_num_weekends = max(max_num_weekends, unassigned_weekends / len(self.doctors))

        #unassigned_nights = len([shift_idx for shift_idx in unassigned_shifts if self.shifts[shift_idx].is_night_shift()])
        #max_num_nights = max(max_num_nights, unassigned_nights / len(self.doctors))

        #unassigned_hours = sum(self.shifts[shift_idx].duration() for shift_idx in unassigned_shifts)
        #max_hours = max(max_hours, unassigned_hours / len(self.doctors))

        # Normalise to roughly 0-1
        preference_score = max(preference_score / len(self.doctors), self._min_dissapointment)
        max_num_weekends = max_num_weekends / self._num_weekend_shifts
        max_hours = max_hours / (self._num_hours / len(self.doctors))
        max_num_nights = max_num_nights / self._num_night_shifts

        unassignment_penalty = len(rota.get_unassigned_shifts()) / len(rota.assignments)
        score = preference_score + max_num_weekends + max_hours + max_num_nights + unassignment_penalty
        return score

    def is_valid_partial_solution(self, rota: Solution) -> bool:
        is_valid = all(
            doctor.is_valid(
                self.get_doctors_rota(rota, doctor_id)
            ) for doctor_id, doctor in enumerate(self.doctors)
        )

        return is_valid

    def generate_candidate_solutions(self, partial_solution: Solution) -> List[Solution]:
        """Generate a new set of unassigned solutions from an existing partial"""

        # get a list of all the unassigned shifts
        unassigned_shifts = partial_solution.get_unassigned_shifts()

        candidates = []
        for shift_id in unassigned_shifts:
            shift = self.shifts[shift_id]
            shifts_to_assign = [shift_id]
            #if shift.is_weekend_shift():
            #    shifts_to_assign = [  # Find the other weekend day
            #        s for s in unassigned_shifts
            #        if shift.time_between(self.shifts[s]) < 24  # it's within a day
            #        and self.shifts[s].is_weekend_shift()
            #        and shift.type == self.shifts[s].type
            #        and shift.date.isoweekday() != self.shifts[s].date.isoweekday()
            #    ]
            if not shifts_to_assign or (len(shifts_to_assign) > 2):
                raise ValueError("Assinging no shifts! or too many!: %d" % len(shifts_to_assign))

            # Try assigning each doctor to said shift
            for doctor_id, doctor in enumerate(self.doctors):

                candidate = partial_solution.assign(doctor_id, shifts_to_assign)
                doctors_shifts = self.get_doctors_rota(candidate, doctor_id)

                # If making this assignment doesn't break any rules, do it.
                if doctor.is_valid(doctors_shifts):
                    candidates.append(candidate)

        return candidates

    def solve(self) -> Solution:

        # Come up with an initial dummy solution to act as an upper bound
        # current_best_solution = heuristic_solve(problem)
        # upper_bound = objective_function(current_best_solution)
        current_best_solution = self.create_empty_solution()
        upper_bound = np.inf
        lower_bound = self._min_dissapointment
        lowest_lower_bound = np.inf  # the lowest one weve seen

        # Populate the set of candidate solutions
        new_candidate_solutions = self.generate_candidate_solutions(current_best_solution)
        total_num_candidates = len(new_candidate_solutions)
        candidate_queue = PriorityQueue()
        candidate_solutions = set()
        deepest = 1
        candidates_scored = 0

        for candidate in new_candidate_solutions:
            score = self.lower_bound(candidate)
            candidate_queue.put(PrioritizedItem(priority=score, inv_depth=1, depth=1, item=candidate))
            candidate_solutions.add(candidate)

        with tqdm(total=total_num_candidates) as pbar:
            while not candidate_queue.empty():
                item = candidate_queue.get()
                candidate_lower_bound = item.priority
                candidate_solution = item.item
                candidate_depth = item.depth

                if not candidate_solution.is_partial():
                    # If we have a concrete rota, score it
                    score = self.score(candidate_solution)
                    candidates_scored += 1

                    # and if it's the best, keep it.
                    # TODO: keep all the options that are equally as good?
                    if score < upper_bound:
                        current_best_solution = candidate_solution
                        upper_bound = score
                        self.current_best = current_best_solution
                        pbar.set_description(f"Scored: {candidates_scored}, Deepest: {deepest}, Lower Bound: {lower_bound}, Current Best: {upper_bound}")

                        if score <= lower_bound:
                            if candidate_solution.is_partial():
                                raise ValueError("how did we get here?")

                            return current_best_solution

                else:
                    # Expand the partial solution
                    new_candidates = self.generate_candidate_solutions(candidate_solution)
                    new_depth = candidate_depth + 1
                    new_inv_depth = 1 / new_depth
                    updated_llb = False

                    for candidate in new_candidates:
                        new_lower_bound = self.lower_bound(candidate)
                        if new_lower_bound < lowest_lower_bound:
                            lowest_lower_bound = new_lower_bound
                            updated_llb = True
                            self.current_llb_solution = candidate

                        if new_lower_bound <= upper_bound and candidate not in candidate_solutions:
                            candidate_queue.put(PrioritizedItem(
                                priority=score, inv_depth=new_inv_depth, depth=new_depth, item=candidate
                            ))
                            candidate_solutions.add(candidate)
                            total_num_candidates += 1

                    if (new_depth > deepest) or updated_llb:
                        deepest = new_depth
                        pbar.set_description(f"Scored: {candidates_scored}, Deepest: {deepest}, Lower Bound: {lower_bound}, Current Best: {upper_bound}, Current LLB: {lowest_lower_bound}")

                    pbar.total = total_num_candidates

                pbar.update(1)

        return current_best_solution

    def calc_minimum_disapointment(self) -> float:
        """Calculates the number of days on which we must dissapoint someone"""

        days_in_rota = list({shift.date for shift in self.shifts})
        days_in_rota.sort()

        clashes = dict()
        for day in days_in_rota:
            todays_shifts = [shift for shift in self.shifts if shift.date == day]
            todays_clashes = [doctor.is_clash(shift) for doctor in self.doctors for shift in todays_shifts]
            clashes[day] = len(todays_clashes)

        # How many days to we have to disapoint everyone on
        num_bad_days = len([v for v in clashes.values() if v == len(self.doctors)])

        # How bad is this dissapointment to each of the doctors
        badness_score = sum([num_bad_days / len(doctor.preferences) for doctor in self.doctors]) / len(self.doctors)
        return badness_score
