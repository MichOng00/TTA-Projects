import tkinter as tk
from tkinter import messagebox, ttk
from wordlebot import WordleSolver, load_word_list

class WordleGUI:
    def __init__(self, master):
        self.master = master
        master.title("Wordle Solver")
        master.geometry("400x550")
        master.resizable(False, False)

        # Apply a clean theme and colors
        style = ttk.Style(master)
        style.theme_use('clam')
        style.configure('TFrame', background = '#c4ba6a')
        style.configure('Guess.TLabel', background='#fafafa', 
                        font=('Helvetica', 20, 'bold'))
        style.configure('TLabel', background='#fafafa', font=('Helvetica', 12))
        style.configure('TButton', font=('Helvetica', 12), padding=6)

        master.configure(bg='#fafafa')

        # Main container
        container = ttk.Frame(master, padding=20, style='TFrame')
        container.grid(sticky='nsew')
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        # Initialize solver
        answers = load_word_list('shuffled_real_wordles.txt')
        allowed = load_word_list('official_allowed_guesses.txt')
        self.solver = WordleSolver(allowed_guesses=allowed, possible_answers=answers)
        self.current_guess = None

        # Guess display
        self.guess_label = ttk.Label(container, text="Click 'Next Guess'", style='Guess.TLabel')
        self.guess_label.grid(row=0, column=0, pady=(0, 15))

        # Feedback entry with real-time validation
        feedback_frame = ttk.Frame(container, style='TFrame')
        feedback_frame.grid(row=1, column=0, pady=(0, 15), sticky='w')
        ttk.Label(feedback_frame, text="Feedback:").grid(row=0, column=0, sticky='e')
        self.feedback_var = tk.StringVar(value='bbbbb')
        self.feedback_var.trace_add('write', self.on_feedback_change)
        self.feedback_entry = tk.Entry(feedback_frame, textvariable=self.feedback_var,
                                       width=7, font=('Courier', 14), bg='#d4fcdc')
        self.feedback_entry.grid(row=0, column=1, padx=(5,0))

        # Next guess button (will be enabled when feedback length==5)
        self.next_button = ttk.Button(container, text="Next Guess", command=self.next_step)
        self.next_button.grid(row=2, column=0, pady=(0, 20))

        # Trigger initial validation
        self.on_feedback_change()

        # Status label
        self.status_label = ttk.Label(container, text=f"Remaining: {len(self.solver.candidates)}", anchor='center')
        self.status_label.grid(row=3, column=0, pady=(0, 10))

        # Candidate list with scrollbar
        list_frame = ttk.Frame(container, style='TFrame')
        list_frame.grid(row=4, column=0, sticky='nsew')
        container.rowconfigure(4, weight=1)

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical')
        self.candidates_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                            font=('Courier', 12), bd=0, highlightthickness=0)
        scrollbar.config(command=self.candidates_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.candidates_listbox.pack(side='left', fill='both', expand=True)
        self.update_candidates_list()

    def on_feedback_change(self, *args):
        s = self.feedback_var.get().lower()
        # Keep only valid chars
        filtered = ''.join(ch for ch in s if ch in 'byg')
        # Enforce max length 5
        if len(filtered) > 5:
            filtered = filtered[:5]
        if filtered != s:
            # avoid recursive trace call issues
            self.feedback_var.trace_remove('write', self.feedback_var.trace_info()[0][1])
            self.feedback_var.set(filtered)
            self.feedback_var.trace_add('write', self.on_feedback_change)
            return
        # Update entry background & Next button state
        if len(filtered) == 5:
            self.feedback_entry.config(bg='#d4fcdc')  # light green
            self.next_button.state(['!disabled'])
        else:
            self.feedback_entry.config(bg='#fcdcdc')  # light red
            self.next_button.state(['disabled'])

    def update_candidates_list(self):
        self.candidates_listbox.delete(0, tk.END)
        if len(self.solver.candidates) <= 10:
            for w in self.solver.candidates:
                self.candidates_listbox.insert(tk.END, w.upper())
        else:
            self.candidates_listbox.insert(tk.END, f"{len(self.solver.candidates)} candidates hidden")

    def next_step(self):
        feedback = self.feedback_var.get().strip().lower()
        # process feedback only if a guess has been made
        if self.current_guess:
            self.solver.filter_feedback(self.current_guess, feedback)
            if feedback == 'ggggg':
                messagebox.showinfo("Solved!", f"Solved with '{self.current_guess.upper()}'!")
                self.master.quit()
                return

        # pick next guess
        next_guess = self.solver.pick_guess()
        if not next_guess:
            messagebox.showerror("Error", "No valid guesses left.")
            self.master.quit()
            return

        self.current_guess = next_guess
        self.solver.guess_history.append(next_guess)
        self.guess_label.config(text=f"Guess: {next_guess.upper()}")
        self.status_label.config(text=f"Remaining: {len(self.solver.candidates)}")
        self.update_candidates_list()

        # reset feedback
        self.feedback_var.set('bbbbb')


def main():
    root = tk.Tk()
    WordleGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()