import numpy as np
from typing import List
from rota_planner.shift import Shift
from rota_planner.doctor import Doctor
from rota_planner.rota import Solution
from tqdm import tqdm


class Problem:

    def __init__(self, shifts: List[Shift], doctors: List[Doctor]):
        self.shifts = shifts
        self.doctors = doctors

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

        for doctor_id, doctor in enumerate(self.doctors):
            doctor_shifts = self.get_doctors_rota(rota, doctor_id)
            doctor_score = doctor.preference_score(doctor_shifts)
            score += doctor_score

        return score

    def lower_bound(self, rota: Solution) -> float:
        """Calculate the best possible score for the partial solution

        We can just use the score, and treat any unassigned shifts as
        not violating any preferences.
        """
        return self.score(rota)

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
            #shift = self.shifts[shift_id]

            # Try assigning each doctor to said shift
            for doctor_id, doctor in enumerate(self.doctors):
                candidate = partial_solution.assign(doctor_id, shift_id)
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
        lower_bound = 0

        # Populate the set of candidate solutions
        candidate_solutions = self.generate_candidate_solutions(current_best_solution)
        total_num_candidates = len(candidate_solutions)

        with tqdm(total=total_num_candidates) as pbar:
            while candidate_solutions:
                candidate_solution = candidate_solutions.pop()

                if not candidate_solution.is_partial():
                    # If we have a concrete rota, score it
                    score = self.score(candidate_solution)

                    # and if it's the best, keep it.
                    # TODO: keep all the options that are equally as good?
                    if score < upper_bound:
                        current_best_solution = candidate_solution
                        upper_bound = score

                        if score <= lower_bound:
                            if candidate_solution.is_partial():
                                raise ValueError("how did we get here?")

                            return current_best_solution

                else:
                    # Expand the partial solution
                    new_candidates = self.generate_candidate_solutions(candidate_solution)
                    filtered_new_candidates = [
                        new_candidate for new_candidate in new_candidates
                        if self.lower_bound(new_candidate) <= upper_bound
                        and new_candidate not in candidate_solutions
                    ]
                    candidate_solutions.extend(filtered_new_candidates)
                    total_num_candidates += len(filtered_new_candidates)
                    pbar.total = total_num_candidates

                pbar.update(1)

        return current_best_solution
