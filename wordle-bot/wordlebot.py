'''
Wordle Solver using the official Wordle word lists
'''

from collections import Counter
import os

class WordleSolver:
    def __init__(self, allowed_guesses, possible_answers):
        self.allowed = allowed_guesses
        self.answers = possible_answers
        self.candidates = list(self.answers)
        self.guess_history = []

    def score_candidates(self):
        # Score letters by frequency
        counts = Counter()
        for w in self.candidates:
            for ch in set(w):
                counts[ch] += 1
        return counts

    def pick_guess(self):
        # If there is only one possible word, guess it
        if len(self.candidates) == 1:
            return self.candidates[0]
        
        # Choose the guess that maximizes coverage of frequent letters
        letter_scores = self.score_candidates()
        best_word = None
        best_score = -1
        for w in self.allowed:
            # Avoid repeating guesses
            if w in self.guess_history:
                continue
            score = sum(letter_scores.get(ch, 0) for ch in set(w))
            if score > best_score:
                best_score = score
                best_word = w
        return best_word
    
    def filter_feedback(self, guess, feedback):
        # feedback: string of length 5 with 'g'= green, 'y'=yellow, 'b' =black
        new_candidates = []

        # count greens & yellows per letter for handling blacks
        requirement = Counter()
        for i, fb in enumerate(feedback):
            if fb in ('g', 'y'):
                requirement[guess[i]] += 1
        
        for w in self.candidates:
            valid = True
            for i, fb in enumerate(feedback):
                ch = guess[i]
                if fb == 'g' and w[i] != ch:
                    valid = False
                    break
                if fb == 'y':
                    if ch == w[i] or ch not in w:
                        valid = False
                        break
                if fb == 'b':
                    if w.count(ch) > requirement[ch]:
                        valid = False
                        break
            if not valid:
                continue
            # Ensure all required counts are met
            for ch, cnt in requirement.items():
                if w.count(ch) < cnt:
                    valid = False
                    break
            if valid:
                new_candidates.append(w)
        self.candidates = new_candidates

    def guess_and_update(self, feedback_callback):
        '''
        Loop until solved or all candidates exhausted
        '''
        while self.candidates:
            guess = self.pick_guess()
            if not guess:
                print("No valid guesses left")
                return None
            self.guess_history.append(guess)
            print(f"Guessing: {guess}")
            fb = feedback_callback(guess)
            if fb == 'ggggg':
                print(f"Solved with '{guess}'!")
                return guess
            self.filter_feedback(guess, fb)
            print(f"Remaining candidates: {len(self.candidates)}")
        print("No candidates left. Something went wrong.")
        return None
    
def load_word_list(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'r') as f:
        return [line.strip() for line in f if len(line.strip()) == 5]
    
def ask_feedback(g): 
    return input(f"Enter feedback for '{g}' (g=green, y=yellow, b=black): ").strip().lower()
    
if __name__ == '__main__':
    answers = load_word_list('shuffled_real_wordles.txt')
    allowed = load_word_list('official_allowed_guesses.txt')
    solver = WordleSolver(allowed_guesses=allowed, possible_answers=answers)
    print(f"Starting with {len(solver.candidates)} answers and {len(solver.allowed)} allowed guesses")
    solver.guess_and_update(ask_feedback)