import tkinter as tk
import os
import pickle

os.chdir(os.path.dirname(__file__))

root = tk.Tk()
root.title("Launcher")
root.geometry("300x400")

def clear_screen():
    for w in root.winfo_children():
        w.pack_forget()

def main_menu():
    clear_screen()
    tk.Label(root, text="Self Driving Car Simulation").pack(pady=10)
    tk.Button(root, text="1. Train a model", command=train_screen).pack(pady=10)
    tk.Button(root, text="2. Run a simulation", command=run_screen).pack(pady=10)

def train_screen():
    clear_screen()
    tk.Label(root, text="Training model...").pack(pady=20)
    
    from training import run_neat
    
    n = 1
    while os.path.exists(f'data/cars/models/model-{n}.bin'):
        n += 1
    f = open(f'data/cars/models/model-{n}.bin', 'wb')
    model = run_neat()
    pickle.dump(model, f)
    f.close()
    
    tk.Label(root, text="Model saved!").pack(pady=10)
    tk.Button(root, text="Back", command=main_menu).pack(pady=10)

control = tk.BooleanVar()

def run_screen():
    clear_screen()
    
    tk.Checkbutton(root, text="Do you want to control a bot?", variable=control).pack(pady=10)
    
    tk.Label(root, text="Number of AI cars:").pack(pady=5)
    num = tk.Entry(root)
    num.pack(pady=5)
    
    error_label = tk.Label(root, text="", fg="red")
    error_label.pack()
    
    def go_next():
        try:
            how_many = int(num.get())
            if how_many <= 0:
                error_label.config(text="Enter a number greater than 0")
                return
        except ValueError:
            error_label.config(text="Please enter a valid number")
            return
        next_page(how_many)
    
    tk.Button(root, text="Next", command=go_next).pack(pady=10)
    tk.Button(root, text="Back", command=main_menu).pack(pady=10)

def next_page(how_many):
    clear_screen()
    
    entries = []
    tk.Label(root, text="Enter model numbers:").pack(pady=10)
    
    for i in range(how_many):
        tk.Label(root, text="Model number for car " + str(i+1)).pack()
        e = tk.Entry(root)
        e.pack(pady=3)
        entries.append(e)
    
    error_label = tk.Label(root, text="", fg="red")
    error_label.pack()
    
    def start_sim():
        genomes = []
        for entry in entries:
            try:
                m = int(entry.get())
                f = open(f'data/cars/models/model-{m}.bin', 'rb')
                model = pickle.load(f)
                genomes.append(model)
                f.close()
            except Exception as e:
                error_label.config(text=f"Error loading model: {e}")
                return
        
        from simulation import run_simulation
        root.destroy()
        run_simulation(genomes, control.get())
    
    tk.Button(root, text="Start Simulation", command=start_sim).pack(pady=10)
    tk.Button(root, text="Back", command=run_screen).pack(pady=10)

main_menu()
root.mainloop()
