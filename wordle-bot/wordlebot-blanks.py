# tinyurl.com/ey5rc5rf

'''
Wordle Solver using the official Wordle word lists
'''

from collections import Counter
import os

class WordleSolver:
    def __init__(self, allowed_guesses, possible_answers):
        pass

    def score_candidates(self):
        # Score letters by frequency
        pass

    def pick_guess(self):
        pass
    
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
        pass
    
def load_word_list(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'r') as f:
        return [line.strip() for line in f if len(line.strip()) == 5]
    
def ask_feedback(g): 
    return input(f"Enter feedback for '{g}' (g=green, y=yellow, b=black): ").strip().lower()
    
if __name__ == '__main__':
    pass